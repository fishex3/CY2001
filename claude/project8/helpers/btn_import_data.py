import streamlit as st
import pandas as pd
import numpy as np
from .feature_helpers import fetch_prices, build_features, get_tvp_var_spillover

def run_import_data(
        selected_tickers,
        price_start,
        study_start,
        study_end,
        vol_window,
        var_conf,
):
    # Safe retrieval of the benchmark from session state
    BENCHMARK = st.session_state.get("BENCHMARK", "^GSPC")

    if True:
        if not selected_tickers:
            st.warning("Please select at least one sector ticker.")
            st.stop()

        with st.spinner("⏳ Downloading price data from yfinance…"):
            all_tickers = selected_tickers + [BENCHMARK]
            prices_all, volume_all = fetch_prices(all_tickers, price_start, study_end)

        sector_px = prices_all[[t for t in selected_tickers if t in prices_all.columns]]
        bm_px = prices_all[BENCHMARK]
        sector_vol = volume_all[[t for t in selected_tickers if t in volume_all.columns]]

        # Filter study window
        mask = (sector_px.index >= pd.Timestamp(study_start)) & (sector_px.index <= pd.Timestamp(study_end))
        sector_px_study = sector_px[mask]
        bm_px_study = bm_px[mask]
        sector_vol_study = sector_vol[mask]
        st.session_state["sector_px_study"] = sector_px_study
        st.session_state["bm_px_study"] = bm_px_study
        st.session_state["sector_vol_study"] = sector_vol_study