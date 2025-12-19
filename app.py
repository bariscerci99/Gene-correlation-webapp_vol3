import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="Gene Spearman Correlation Explorer", layout="wide")
st.title("Gene–Gene Spearman Correlation Explorer (with clinical filtering)")

st.write(
    "Upload:\n"
    "1) **Expression matrix (.csv.gz)**: Column 1 = gene names; other columns = sample IDs.\n"
    "2) **Clinical table (.csv)**: Each row = patient/sample. Filter patients (continuous vars by quartiles), then run correlations."
)

# -----------------------------
# Utilities
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

def is_continuous(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s):
        return False
    return s.dropna().nunique() >= 10  # heuristic

def quartile_bins(x: pd.Series) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    valid = x_num.notna()
    out = pd.Series(index=x.index, dtype="float")
    if valid.sum() == 0:
        return out

    q1 = x_num[valid].quantile(0.25)
    q2 = x_num[valid].quantile(0.50)
    q3 = x_num[valid].quantile(0.75)

    out.loc[valid & (x_num <= q1)] = 1
    out.loc[valid & (x_num > q1) & (x_num <= q2)] = 2
    out.loc[valid & (x_num > q2) & (x_num <= q3)] = 3
    out.loc[valid & (x_num > q3)] = 4
    return out

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

    res = pd.DataFrame({"Gene": genes, "Spearman_rho": rhos, "p_value": pvals, "N_pairs": n_pairs})

    valid = res["p_value"].notna()
    qvals = np.full(res.shape[0], np.nan, dtype=float)
    if valid.sum() > 0:
        qvals[valid.values] = bh_qvalues(res.loc[valid, "p_value"].to_numpy(dtype=float))
    res["q_value"] = qvals

    res["abs_rho"] = res["Spearman_rho"].abs()
    res = res.sort_values(["abs_rho", "p_value"], ascending=[False, True]).drop(columns="abs_rho")
    return res

# -----------------------------
# Wizard state
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

def go(step: int):
    st.session_state.step = step

# -----------------------------
# STEP 1: Uploads
# -----------------------------
st.header("Step 1 — Upload files")

expr_file = st.file_uploader("1) Upload expression matrix (.csv.gz only)", type=["gz"], key="expr_up")
clin_file = st.file_uploader("2) Upload clinical table (.csv) (optional but needed for filtering)", type=["csv"], key="clin_up")

colA, colB = st.columns([1, 1])
with colA:
    if st.button("Next →", disabled=(expr_file is None)):
        go(2)

if expr_file is None:
    st.stop()

# Load expression early (cached) so later steps are fast/stable
df_expr = load_expression_gz(expr_file.getvalue())
gene_col = df_expr.columns[0]
expr_samples = pd.Index(df_expr.columns[1:]).astype(str)

st.caption(f"Expression loaded • Genes: {df_expr.shape[0]:,} • Samples: {len(expr_samples):,}")

if st.session_state.step < 2:
    st.stop()

# -----------------------------
# STEP 2: Clinical mapping (if provided)
# -----------------------------
st.header("Step 2 — Clinical mapping (optional)")

df_clin = None
id_col = None
overlap_samples = expr_samples

if clin_file is None:
    st.info("No clinical file uploaded. You can still run correlations on all samples.")
    if st.button("Next →"):
        go(4)  # skip to gene/correlation step
    st.stop()

df_clin = load_clinical_csv(clin_file.getvalue()).copy()

id_col = st.selectbox(
    "Select the clinical column that contains sample IDs matching expression columns",
    options=list(df_clin.columns),
    key="id_col",
)

df_clin[id_col] = df_clin[id_col].astype(str)
overlap = pd.Index(df_clin[id_col]).intersection(expr_samples)
st.caption(f"Overlapping samples (clinical ∩ expression): {len(overlap):,}")

if len(overlap) == 0:
    st.error("No overlapping sample IDs between clinical table and expression matrix.")
    st.stop()

if st.button("Next →"):
    go(3)

if st.session_state.step < 3:
    st.stop()

# -----------------------------
# STEP 3: Choose variables to filter on (no filtering applied yet)
# -----------------------------
st.header("Step 3 — Choose clinical variables to filter on")

candidate_vars = [c for c in df_clin.columns if c != id_col]
chosen_vars = st.multiselect(
    "Select variables to filter on (you'll configure filters in the next step)",
    options=candidate_vars,
    default=st.session_state.get("chosen_vars", []),
    key="chosen_vars",
)

if st.button("Next →"):
    go(4)

if st.session_state.step < 4:
    st.stop()

# -----------------------------
# STEP 4: Configure filters + apply (still no correlation run)
# -----------------------------
st.header("Step 4 — Configure filters and apply to patients")

# Restrict clinical to overlapping samples
df_clin2 = df_clin[df_clin[id_col].isin(overlap)].reset_index(drop=True).copy()

filters = st.session_state.get("filters", {})
new_filters = {}

st.write("Set filters below, then click **Apply filters →**. The correlation analysis will not run until Step 6.")

for var in chosen_vars:
    s = df_clin2[var]

    if is_continuous(s):
        st.markdown(f"**{var}** (continuous) — choose quartiles")
        prev = filters.get(var, {"type": "quartiles", "quartiles": [1, 2, 3, 4]})
        q_choice = st.multiselect(
            f"Quartiles for {var} (Q1=lowest, Q4=highest)",
            options=[1, 2, 3, 4],
            default=prev.get("quartiles", [1, 2, 3, 4]),
            key=f"q_{var}",
        )
        new_filters[var] = {"type": "quartiles", "quartiles": q_choice}
    else:
        st.markdown(f"**{var}** (categorical) — choose allowed values")
        vals = s.astype("string")
        uniq = sorted([str(u) for u in vals.dropna().unique().tolist()])
        has_na = vals.isna().any()
        options = uniq + (["(missing)"] if has_na else [])
        prev = filters.get(var, {"type": "categorical", "values": options})
        picked = st.multiselect(
            f"Keep values for {var}",
            options=options,
            default=prev.get("values", options),
            key=f"c_{var}",
        )
        new_filters[var] = {"type": "categorical", "values": picked}

apply_clicked = st.button("Apply filters →")

if apply_clicked:
    st.session_state.filters = new_filters

    keep = pd.Series(True, index=df_clin2.index)

    for var, spec in new_filters.items():
        s = df_clin2[var]
        if spec["type"] == "quartiles":
            bins = quartile_bins(s)
            keep &= bins.isin(spec["quartiles"]).fillna(False)
        else:
            vals = s.astype("string")
            picked = spec["values"]
            m = pd.Series(False, index=df_clin2.index)
            if "(missing)" in picked:
                m |= vals.isna()
            picked_non_missing = [p for p in picked if p != "(missing)"]
            if picked_non_missing:
                m |= vals.fillna("").isin(picked_non_missing)
            keep &= m

    kept_ids = pd.Index(df_clin2.loc[keep, id_col].astype(str))
    filtered_samples = expr_samples.intersection(kept_ids)

    st.session_state.filtered_samples = list(filtered_samples)
    st.success(f"Filters applied. Patients kept: {len(filtered_samples):,}")

# Show current filtered count (if applied)
current_filtered = st.session_state.get("filtered_samples", None)
if current_filtered is None:
    st.info("No filters applied yet in this step. Click **Apply filters →** to finalize the patient subset.")
else:
    st.caption(f"Current filtered patients: {len(current_filtered):,}")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Back ←"):
        go(3)
with col2:
    # allow next if filters are applied OR no chosen vars (i.e., overlap-only)
    can_next = (current_filtered is not None) or (len(chosen_vars) == 0)
    if st.button("Next →", disabled=not can_next):
        go(5)

if st.session_state.step < 5:
    st.stop()

# -----------------------------
# STEP 5: Select gene + settings
# -----------------------------
st.header("Step 5 — Choose gene and settings (analysis still not running)")

if clin_file is None:
    filtered_samples = list(expr_samples)
else:
    if len(chosen_vars) == 0:
        filtered_samples = list(overlap)  # overlap-only
    else:
        filtered_samples = st.session_state.get("filtered_samples", [])
        if not filtered_samples:
            st.error("No patients available after filtering. Go back and relax filters.")
            st.stop()

# Keep expression columns in original order
filtered_samples_ordered = [c for c in df_expr.columns[1:] if str(c) in set(filtered_samples)]
if len(filtered_samples_ordered) < 3:
    st.error("Too few patients to compute correlations. Go back and relax filters.")
    st.stop()

df_expr_filt = df_expr[[gene_col] + filtered_samples_ordered].copy()
genes = df_expr_filt[gene_col].astype(str).tolist()

st.caption(f"Patients used for correlation: {len(filtered_samples_ordered):,}")

selected_gene = st.selectbox("Search and select a gene", options=genes, key="sel_gene")
min_pairs = st.number_input(
    "Minimum paired samples (non-missing) required per correlation",
    min_value=3,
    max_value=5000,
    value=min(10, len(filtered_samples_ordered)),
    step=1,
    key="min_pairs",
)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Back ←"):
        go(4)
with col2:
    if st.button("Next →"):
        go(6)

if st.session_state.step < 6:
    st.stop()

# -----------------------------
# STEP 6: Run analysis (only here)
# -----------------------------
st.header("Step 6 — Run genome-wide correlation")

run = st.button("Run correlation analysis now")

if not run:
    st.stop()

with st.spinner("Computing Spearman correlations…"):
    res = compute_spearman_all(
        df_expr=df_expr_filt,
        gene_col=gene_col,
        selected_gene=st.session_state.sel_gene,
        min_pairs=int(st.session_state.min_pairs),
    )

st.success(f"Done. Computed correlations for {res.shape[0]:,} genes.")
st.dataframe(res, use_container_width=True, height=600)

st.download_button(
    "Download results CSV",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name=f"{st.session_state.sel_gene}_spearman_correlations_filtered.csv",
    mime="text/csv",
)

if st.button("Start over"):
    for k in ["step", "filters", "filtered_samples", "chosen_vars", "id_col", "sel_gene", "min_pairs"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.step = 1
    st.rerun()
