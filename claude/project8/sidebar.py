import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, date

from helpers.load_on_run import load_on_run

def create_sidebar():
    SECTORS = st.session_state["SECTORS"]
    var = {}
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        st.markdown("---")

        st.markdown("### 📅 Data Parameters")
        var["price_start"]  = st.date_input("Price history start", value=date(2005, 1, 1), min_value=date(2000,1,1))
        var["study_start"]  = st.date_input("Study window start",  value=date(2010, 1, 1), min_value=date(2005,1,1))
        var["study_end"]    = st.date_input("Study window end",    value=date.today())
        var["selected_tickers"] = st.multiselect(
            "Sectors", list(SECTORS.keys()), default=list(SECTORS.keys()),
            format_func=lambda t: f"{t} – {SECTORS[t]}"
        )


        st.markdown("---")
        st.markdown("### 📐 Feature Parameters")
        var["vol_window_feature_parameter"]   = st.slider("Rolling window (days)", 10, 120, 30, 5)
        var["var_conf"]     = st.slider("VaR confidence (%)", 90, 99, 95)


        st.markdown("---")
        st.markdown("### 🌐 TVP-VAR Parameters")
        var["vol_window_tvp_var"] = st.slider("Volatility window (days)", 10, 126, 30)
        var["forecast_h"]   = st.slider("Forecast horizon (days)", 5, 30, 10)
        var["kappa1"]       = st.slider("Forgetting factor κ₁", 0.90, 0.999, 0.99, 0.001, format="%.3f")
        var["kappa2"]       = st.slider("Forgetting factor κ₂", 0.90, 0.999, 0.99, 0.001, format="%.3f")


        st.markdown("---")
        st.markdown("### 🤖 ML Parameters")
        var["n_cv_folds"]   = st.slider("CV folds", 3, 10, 5)
        var["test_size"]    = st.slider("Test quarters per fold", 2, 8, 4)

        with st.expander("XGBoost"):
            var["xgb_n_est"]  = st.slider("n_estimators",  50, 800, 300, 50)
            var["xgb_depth"]  = st.slider("max_depth",       1,   8,   2)
            var["xgb_lr"]     = st.slider("learning_rate", 0.01, 0.30, 0.05, 0.01)
            var["xgb_sub"]    = st.slider("subsample",      0.5, 1.0, 0.8, 0.05)
            var["xgb_col"]    = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)
            var["xgb_alpha"]  = st.slider("reg_alpha",      0.0, 2.0, 0.5, 0.1)
            var["xgb_lambda"] = st.slider("reg_lambda",     0.0, 5.0, 2.0, 0.5)
            var["xgb_mcw"]    = st.slider("min_child_weight", 1, 20, 5)

        with st.expander("LightGBM"):
            var["lgb_n_est"]  = st.slider("n_estimators (LGB)", 50, 800, 300, 50)
            var["lgb_depth"]  = st.slider("max_depth (LGB)",      1,   8,   4)
            var["lgb_lr"]     = st.slider("learning_rate (LGB)", 0.01, 0.30, 0.05, 0.01)

        with st.expander("SVM / MLP / LR"):
            var["svm_c"]       = st.slider("SVM C",         0.01, 10.0, 1.0, 0.1)
            var["mlp_layers"]  = st.text_input("MLP hidden layers", "(32,16)")
            var["lr_c"]        = st.slider("LR C",          0.01, 10.0, 0.5, 0.1)

        var["selected_features"] = st.multiselect(
            "Features to use",
            ["volatility","downside_volatility","beta","market_correlation",
             "amihud_illiquidity","volume","var","skewness","kurtosis",
             "log_market_cap","average_sector_correlation"],
            default=["volatility","downside_volatility","beta","market_correlation",
                     "amihud_illiquidity","volume","var","skewness","kurtosis",
                     "log_market_cap","average_sector_correlation"],
        )

        run_btn = st.button("🚀 Run Full Pipeline", type="primary", width="stretch")
        import_data_btn = st.button("Import data", type="primary", width="stretch")
        if import_data_btn:
            load_on_run(
                var["selected_tickers"],
                var["price_start"],
                var["study_start"],
                var["study_end"],
                var["vol_window_tvp_var"],
                var["var_conf"],
                var["forecast_h"],
                var["kappa1"],
                var["kappa2"]
            )
        calculate_spillover_btn = st.button("Calculate spillover", type="primary", width="stretch")
        return var