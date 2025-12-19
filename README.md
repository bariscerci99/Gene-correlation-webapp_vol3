# Gene Correlation Web App (Streamlit)

Upload a wide gene expression matrix (genes x samples), search/select a gene, and compute Spearman correlations vs all other genes with p-values and Benjamini–Hochberg (BH) q-values.

## Input format

A CSV where:

- **Column 1** = gene identifier (e.g., `Hugo_Symbol`)
- **Remaining columns** = sample IDs (e.g., `MB-0362`, `MB-0346`, ...)

Each row is a gene; values are expression levels.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (must include `requirements.txt`).
2. Create a Streamlit Cloud app pointing to this repo.
3. Set the main file path to `app.py`.
4. Reboot the app after any changes.

## Output

A downloadable CSV with columns:

- `Gene`
- `Spearman_rho`
- `p_value`
- `N_pairs`
- `q_value`
