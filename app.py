import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="Gene Spearman Correlation Explorer", layout="wide")
st.title("Gene–Gene Spearman Correlation Explorer")

st.write(
    "Upload:\n"
    "1) **Expression matrix (.csv.gz)**: Column 1 = gene names; other columns = sample IDs.\n"
    "2) **Clinical table (.csv)**: Each row = patient/sample. You can filter patients using clinical variables\n"
    "   (continuous variables can be filtered by quartiles). Correlations are computed on the filtered patients only.\n\n"
    "Note: If your expression file contains duplicate gene names, duplicates are collapsed by **mean**."
)

# -----------------------------
# Uploaders
# -----------------------------
expr_file = st.file_uploader("1) Upload expression matrix (.csv.gz only)", type=["gz"])
clin_file = st.file_uploader("2) Upload clinical table (.csv)", type=["csv"])

# -----------------------------
# Helpers
# -----------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    return multipletests(pvals, method="fdr_bh")[1]

@st.cache_data(show_spinner=False)
def load_expression_gz(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), compression="gzip")

    gene_col = df.columns[0]
    df[gene_col] = df[gene_col].astype(str)

    sample_cols = df.columns[1:]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce")

    # Collapse duplicate genes by mean across samples
    df = df.groupby(gene_col, as_index=False)[sample_cols].mean()
    return df

@st.cache_data(show_spinner=False)
def load_clinical_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def compute_spearman_all(df_expr: pd.DataFrame, gene_col: str, selected_gene: str, min_pairs: int = 10) -> pd.DataFrame:
    sample_cols = df_expr.columns[1:]
    mat = df_expr.set_index(gene_col)

    if selected_gene not in mat.index:
        raise ValueError(f"Gene '{selected_gene}' not found in matrix.")

    target = np.asarray(mat.loc[selected_gene, sample_cols], dtype=float).ravel()

    genes = mat.index.to_list()
    rhos, pvals, n_pairs = [], [], []

    for g in genes:
        x = np.asarray(mat.loc[g, sample_cols], dtype=float).ravel()
        mask = np.isfinite(target) & np.isfinite(x)
        n = int(mask.sum())
        n_pairs.append(n)

        if g == selected_gene or n < min_pairs:
            rhos.append(np.nan)
            pvals.append(np.nan)
            continue

        rho, p = spearmanr(target[mask], x[mask])
        rhos.append(float(rho))
        pvals.append(float(p))

    res = pd.DataFrame({
        "Gene": genes,
        "Spearman_rho": rhos,
        "p_value": pvals,
        "N_pairs": n_pairs,
    })

    valid = res["p_value"].notna()
    qvals = np.full(res.shape[0], np.nan, dtype=float)
    if valid.sum() > 0:
        qvals[valid.values] = bh_qvalues(res.loc[valid, "p_value"].to_numpy(dtype=float))
    res["q_value"] = qvals

    res["abs_rho"] = res["Spearman_rho"].abs()
    res = res.sort_values(["abs_rho", "p_value"], ascending=[False, True]).drop(columns="abs_rho")
    return res

def is_continuous(s: pd.Series) -> bool:
    # Consider numeric with enough unique values as continuous
    if not pd.api.types.is_numeric_dtype(s):
        return False
    nun = s.dropna().nunique()
    return nun >= 10  # heuristic

def quartile_mask(x: pd.Series, chosen_quartiles: list[int]) -> pd.Series:
    """
    Returns boolean mask selecting rows whose x falls into chosen quartiles.
    Quartiles are computed on non-missing values.
    """
    x_num = pd.to_numeric(x, errors="coerce")
    valid = x_num.notna()
    if valid.sum() == 0:
        return pd.Series([False] * len(x), index=x.index)

    q1 = x_num[valid].quantile(0.25)
    q2 = x_num[valid].quantile(0.50)
    q3 = x_num[valid].quantile(0.75)

    # Define quartile bins: Q1 lowest, Q4 highest
    q_bin = pd.Series(index=x.index, dtype="float")
    q_bin.loc[valid & (x_num <= q1)] = 1
    q_bin.loc[valid & (x_num > q1) & (x_num <= q2)] = 2
    q_bin.loc[valid & (x_num > q2) & (x_num <= q3)] = 3
    q_bin.loc[valid & (x_num > q3)] = 4

    return q_bin.isin(chosen_quartiles).fillna(False)

# -----------------------------
# Guardrails
# -----------------------------
if expr_file is None:
    st.info("Upload the expression matrix (.csv.gz) to begin.")
    st.stop()

df_expr = load_expression_gz(expr_file.getvalue())
gene_col = df_expr.columns[0]

expr_samples = pd.Index(df_expr.columns[1:]).astype(str)

# -----------------------------
# Clinical alignment + filtering UI
# -----------------------------
filtered_samples = expr_samples.copy()
active_filters_summary = []

if clin_file is not None:
    df_clin = load_clinical_csv(clin_file.getvalue())

    st.subheader("Clinical table mapping")
    with st.expander("Set the sample/patient ID column", expanded=True):
        id_col = st.selectbox(
            "Which clinical column contains sample IDs that match the expression columns?",
            options=list(df_clin.columns),
            index=0,
        )

    df_clin = df_clin.copy()
    df_clin[id_col] = df_clin[id_col].astype(str)

    # Restrict to overlapping samples
    overlap = pd.Index(df_clin[id_col]).intersection(expr_samples)
    if len(overlap) == 0:
        st.error("No overlapping sample IDs between the clinical table and expression matrix.")
        st.stop()

    df_clin = df_clin[df_clin[id_col].isin(overlap)].reset_index(drop=True)
    st.caption(f"Overlapping samples (clinical ∩ expression): {len(overlap):,}")

    st.subheader("Filter patients using clinical variables")
    # Choose variables to filter on
    candidate_vars = [c for c in df_clin.columns if c != id_col]
    chosen_vars = st.multiselect(
        "Choose clinical variables to filter on",
        options=candidate_vars,
        default=[],
    )

    # Build mask across clinical rows
    keep = pd.Series(True, index=df_clin.index)

    for var in chosen_vars:
        col = df_clin[var]

        # Continuous numeric -> quartile filter
        if is_continuous(col):
            st.markdown(f"**{var}** (continuous) — filter by quartiles")
            q_choice = st.multiselect(
                f"Select quartiles for {var}",
                options=[1, 2, 3, 4],
                default=[1, 2, 3, 4],
                key=f"quart_{var}",
                help="Q1=lowest values, Q4=highest values (computed among non-missing).",
            )
            m = quartile_mask(col, q_choice)
            keep &= m
            active_filters_summary.append(f"{var}: quartiles {q_choice}")
        else:
            # Treat as categorical
            st.markdown(f"**{var}** (categorical) — filter by values")
            # Use string categories; keep NaN as its own option if present
            vals = col.astype("string")
            uniq = vals.dropna().unique().tolist()
            uniq_sorted = sorted([str(u) for u in uniq])

            # If NaNs exist, offer "(missing)" option
            has_na = vals.isna().any()
            options = uniq_sorted + (["(missing)"] if has_na else [])

            default = options  # keep all by default
            picked = st.multiselect(
                f"Keep values for {var}",
                options=options,
                default=default,
                key=f"cat_{var}",
            )

            m = pd.Series(False, index=df_clin.index)
            if "(missing)" in picked:
                m |= vals.isna()
            picked_non_missing = [p for p in picked if p != "(missing)"]
            if picked_non_missing:
                m |= vals.fillna("").isin(picked_non_missing)

            keep &= m
            active_filters_summary.append(f"{var}: {picked[:5]}{'…' if len(picked) > 5 else ''}")

    # Apply mask
    kept_ids = pd.Index(df_clin.loc[keep, id_col].astype(str))
    filtered_samples = expr_samples.intersection(kept_ids)

    st.markdown("### Filter summary")
    if active_filters_summary:
        st.write("- " + "\n- ".join(active_filters_summary))
    else:
        st.write("No clinical filters applied (using all overlapping patients).")

    st.caption(f"Patients kept after filtering: {len(filtered_samples):,}")
else:
    st.info("Optional: upload a clinical table to enable patient filtering. Using all samples in the expression matrix.")

# -----------------------------
# Subset expression matrix to filtered samples
# -----------------------------
if len(filtered_samples) < 3:
    st.error("Too few patients after filtering to compute correlations. Relax filters and try again.")
    st.stop()

# Keep expression columns in the original order
filtered_samples = [s for s in df_expr.columns[1:] if str(s) in set(filtered_samples)]
df_expr_filt = df_expr[[gene_col] + filtered_samples].copy()

# -----------------------------
# Gene selection + run
# -----------------------------
genes = df_expr_filt[gene_col].astype(str).tolist()
st.subheader("Search and select a gene")
selected_gene = st.selectbox("Type to search", options=genes, index=0)

with st.sidebar:
    st.header("Correlation settings")
    min_pairs = st.number_input(
        "Minimum paired samples",
        min_value=3,
        max_value=5000,
        value=min(10, len(filtered_samples)),
        step=1,
    )

if not st.button("Compute genome-wide correlations on filtered patients"):
    st.stop()

with st.spinner("Computing Spearman correlations…"):
    res = compute_spearman_all(
        df_expr=df_expr_filt,
        gene_col=gene_col,
        selected_gene=selected_gene,
        min_pairs=int(min_pairs),
    )

st.success(f"Computed correlations for {res.shape[0]:,} genes using {len(filtered_samples):,} patients.")
st.dataframe(res, use_container_width=True, height=600)

st.download_button(
    "Download results CSV",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_gene}_spearman_correlations_filtered.csv",
    mime="text/csv",
)


