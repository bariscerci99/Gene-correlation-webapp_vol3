import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="Gene Spearman Correlation Explorer", layout="wide")
st.title("Gene–Gene Spearman Correlation Explorer")

st.write(
    "Upload a **gzip-compressed expression matrix (.csv.gz)** where:\n"
    "- **Column 1** = gene names (e.g., Hugo_Symbol)\n"
    "- **Other columns** = samples\n\n"
    "Then search for a gene and compute Spearman correlations against all other genes."
)

# 🔒 Only allow csv.gz
expr_file = st.file_uploader(
    "Upload expression matrix (.csv.gz only)",
    type=["gz"]
)

@st.cache_data(show_spinner=False)
def load_expression_gz(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), compression="gzip")
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    return df

def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    return multipletests(pvals, method="fdr_bh")[1]

def compute_spearman_all(
    df: pd.DataFrame,
    gene_col: str,
    selected_gene: str,
    min_pairs: int = 10,
) -> pd.DataFrame:

    sample_cols = df.columns[1:]
    mat = df.set_index(gene_col)

    if selected_gene not in mat.index:
        raise ValueError(f"Gene '{selected_gene}' not found in matrix.")

    target = mat.loc[selected_gene, sample_cols].to_numpy(dtype=float)

    genes = mat.index.to_list()
    rhos, pvals, n_pairs = [], [], []

    for g in genes:
        x = mat.loc[g, sample_cols].to_numpy(dtype=float)
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
    qvals = np.full(res.shape[0], np.nan)
    if valid.sum() > 0:
        qvals[valid.values] = bh_qvalues(res.loc[valid, "p_value"].values)

    res["q_value"] = qvals

    res["abs_rho"] = res["Spearman_rho"].abs()
    res = res.sort_values(
        ["abs_rho", "p_value"],
        ascending=[False, True]
    ).drop(columns="abs_rho")

    return res


if expr_file is None:
    st.info("Please upload a `.csv.gz` file to begin.")
    st.stop()

df = load_expression_gz(expr_file.getvalue())

gene_col = df.columns[0]
genes = df[gene_col].tolist()

st.caption(
    f"Genes: {df.shape[0]:,} • Samples: {df.shape[1]-1:,}"
)

st.subheader("Search and select a gene")
selected_gene = st.selectbox(
    "Type to search",
    options=genes,
)

with st.sidebar:
    st.header("Settings")
    min_pairs = st.number_input(
        "Minimum paired samples",
        min_value=3,
        max_value=5000,
        value=10,
        step=1,
    )

if not st.button("Compute correlations"):
    st.stop()

with st.spinner("Computing Spearman correlations…"):
    res = compute_spearman_all(
        df=df,
        gene_col=gene_col,
        selected_gene=selected_gene,
        min_pairs=int(min_pairs),
    )

st.success(f"Computed correlations for {res.shape[0]:,} genes.")
st.dataframe(res, use_container_width=True, height=600)

st.download_button(
    "Download results CSV",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_gene}_spearman_correlations.csv",
    mime="text/csv",
)
