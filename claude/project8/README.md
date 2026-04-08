# Project 8 — Sector Spillovers Dashboard (Streamlit)

Interactive Streamlit dashboard for analyzing U.S. sector ETF dynamics:

- Download and visualize **prices/returns** (via `yfinance`)
- Compute a set of **risk/liquidity features** (volatility, beta, VaR, Amihud illiquidity, correlations, moments, etc.)
- Estimate **TVP-VAR connectedness / spillovers** using an R implementation (ConnectednessApproach)
- Train and compare **ML classifiers** (XGBoost, LightGBM, SVM, MLP, Logistic Regression) to classify sectors as **transmitters vs receivers**

## What’s in here

- **`app.py`**: Streamlit entrypoint (tabs UI)
- **`sidebar.py`**: Sidebar controls and “Run full analysis” trigger
- **`helpers/`**:
  - `load_on_run.py`: Full pipeline orchestration (fetch data → features → TVP-VAR → panel build)
  - `feature_helpers.py`: Feature engineering + ML helpers
- **`tvp_var_spillover.py`**: Python wrapper that calls **`Rscript`** to run TVP-VAR spillovers
- **`r_packages/tvp_var_spillover.R`**: R script that runs `ConnectednessApproach(..., model="TVP-VAR")`
- **`tabs/`**: Individual Streamlit tabs (plots, pipeline, comparisons)

## Prerequisites

### Python
- Python 3.10+ recommended
- Install Python dependencies from `requirements.txt`

### R (required for spillover step)
The spillover analysis uses `Rscript` to run `r_packages/tvp_var_spillover.R`.

- Install **R** (so `Rscript` is available)
- Install required R packages (see below)

If `Rscript` is not on your PATH, you can set an environment variable named `RSCRIPT` pointing to it (example below).

## Setup (Windows / PowerShell)

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Install R packages

In an R console (or RStudio), run:

```r
install.packages(c(
  "ConnectednessApproach",
  "ggplot2",
  "lubridate",
  "dplyr",
  "tidyr",
  "zoo",
  "xts"
))
```

## Running the app

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then in the sidebar:

- Choose dates, tickers (sector ETFs), and parameters
- Click **Run full analysis**

Notes:

- The **TVP-VAR spillover** step can take **~5 minutes** depending on the window size and machine.
- Data is downloaded at runtime from Yahoo Finance via `yfinance`.

## Rscript configuration (optional)

If Streamlit can’t find `Rscript`, set the `RSCRIPT` environment variable.

Example:

```powershell
$env:RSCRIPT = "C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
streamlit run app.py
```

## Troubleshooting

### “Could not find Rscript on PATH”
- Install R, or set `RSCRIPT` as shown above.

### Spillover step fails with an R error
- Confirm the required R packages are installed.
- Try running the R script directly to see the full error:

```powershell
Rscript .\r_packages\tvp_var_spillover.R --help
```

### yfinance download issues
- Retry later (rate limits happen), or reduce the date range / number of tickers.

