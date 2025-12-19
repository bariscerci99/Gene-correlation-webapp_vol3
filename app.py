import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="Gene Spearman Correlation Explorer", layout="wide")
st.title("Gene–Gene Spearman correlation (with BH q-values)")

st.write(
    "Upload an expression matrix where:\n"
    "- **Column 1** = gene names (e.g., Hugo_Symbol)\n"
    "- **Other columns** = samples (e.g., MB-0362, MB-0346, ...)\n\n"
    "Then search/select a gene and compute Spearman correlations vs all other genes."
)

expr_file = st.file_uploader("Upload expression matrix (.csv)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_expression(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Ensure first column treated as gene identifier
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    return df

def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    # multipletests returns adjusted p-values in same order
    return multipletests(pvals, method="fdr_bh")[1]

def compute_spearman_all(
    df: pd.DataFrame,
    gene_col: str,
    selected_gene: str,
    min_pairs: int = 10,
) -> pd.DataFrame:
    """Compute Spearman correlation of selected_gene vs all other genes across samples."""
    sample_cols = df.columns[1:]
    tmp = df.set_index(gene_col)

    if selected_gene not in tmp.index:
        raise ValueError(f"Gene '{selected_gene}' not found.")

    target = tmp.loc[selected_gene, sample_cols].to_numpy(dtype=float)

    genes = tmp.index.to_list()
    rhos = []
    pvals = []
    n_pairs_list = []

    for g in genes:
        x = tmp.loc[g, sample_cols].to_numpy(dtype=float)
        mask = np.isfinite(target) & np.isfinite(x)
        n_pairs = int(mask.sum())
        n_pairs_list.append(n_pairs)

        # Skip self and low-pair genes
        if n_pairs < min_pairs or g == selected_gene:
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
        "N_pairs": n_pairs_list,
    })

    # BH correction on valid p-values only
    valid = res["p_value"].notna()
    qvals = np.full(res.shape[0], np.nan, dtype=float)
    if valid.sum() > 0:
        qvals[valid.to_numpy()] = bh_qvalues(res.loc[valid, "p_value"].to_numpy(dtype=float))
    res["q_value"] = qvals

    # Sort by |rho| then p-value (NaNs go to bottom automatically)
    res["abs_rho"] = res["Spearman_rho"].abs()
    res = res.sort_values(["abs_rho", "p_value"], ascending=[False, True]).drop(columns=["abs_rho"])
    return res

if expr_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = load_expression(expr_file.getvalue())
gene_col = df.columns[0]
sample_cols = df.columns[1:]

st.caption(f"Detected gene column: `{gene_col}` • Genes: {df.shape[0]:,} • Samples: {len(sample_cols):,}")

genes = df[gene_col].astype(str).tolist()

st.subheader("Search and select a gene")
selected_gene = st.selectbox(
    "Type to search (supports partial search)",
    options=genes,
    index=0,
)

with st.sidebar:
    st.header("Settings")
    min_pairs = st.number_input("Min non-missing sample pairs", min_value=3, max_value=5000, value=10, step=1)

run = st.button("Compute correlations")

if not run:
    st.stop()

with st.spinner("Computing Spearman correlations…"):
    res = compute_spearman_all(df=df, gene_col=gene_col, selected_gene=selected_gene, min_pairs=int(min_pairs))

st.success(f"Done. Results for {res.shape[0]:,} genes (self-correlation is NaN).")
st.dataframe(res, use_container_width=True, height=600)

csv_bytes = res.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results CSV",
    data=csv_bytes,
    file_name=f"{selected_gene}_spearman_correlations.csv",
    mime="text/csv",
)
