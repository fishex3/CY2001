import streamlit as st
import plotly.express as px
from sidebar import create_sidebar

from tabs.tab_ticker_features import render_ticker_features
from tabs.tab_feature_engineering import render_feature_engineering
from tabs.tab_spillover_analysis import render_spillover_analysis
from tabs.tab_ml_pipeline import render_ml_pipeline
from tabs.tab_model_comparison import render_model_comparison


def main():
    SECTORS = {
        "XLF": "Financials",    "XLE": "Energy",          "XLK": "Technology",
        "XLV": "Healthcare",    "XLI": "Industrials",     "XLY": "Cons. Discretionary",
        "XLP": "Cons. Staples", "XLB": "Materials",       "XLU": "Utilities",
    }
    ETF_MARKET_CAP = {
        "XLF": 45e9, "XLE": 38e9, "XLK": 65e9, "XLV": 42e9,
        "XLI": 18e9, "XLY": 22e9, "XLP": 18e9, "XLB": 12e9, "XLU": 14e9,
    }
    BENCHMARK = "SPY"
    PALETTE = px.colors.qualitative.Set2
    st.session_state["SECTORS"] = SECTORS
    st.session_state["ETF_MARKET_CAP"] = ETF_MARKET_CAP
    st.session_state["BENCHMARK"] = BENCHMARK
    st.session_state["PALETTE"] = PALETTE
    var = create_sidebar()
    st.session_state.update(var)
    tabs = st.tabs([
        "📈 Price & Returns",
        "🔬 Feature Engineering",
        "🌊 Spillover Analysis",
        "🤖 ML Pipeline",
        "📊 Model Comparison",
    ])
    if "sector_px" in st.session_state:
        with tabs[0]:

            render_ticker_features(
                sector_px=st.session_state["sector_px"],
                tickers=st.session_state["selected_tickers"],
            )
        with tabs[1]:
            render_feature_engineering(
                calculated_features=st.session_state["calculated_features"],
                tickers=st.session_state["selected_tickers"],
                selected_features=st.session_state["selected_features"],
                panel_df=st.session_state["panel_df"],
            )
        with tabs[2]:
            render_spillover_analysis(
                tci_df=st.session_state["tci_df"],
                net_q=st.session_state["net_q"],
                tickers=st.session_state["selected_tickers"],
                panel_df=st.session_state["panel_df"],
            )
        with tabs[3]:
            render_ml_pipeline(
                selected_features=st.session_state["selected_features"],
                panel_df=st.session_state["panel_df"],
                n_cv_folds=st.session_state["n_cv_folds"],
                test_size=st.session_state["test_size"],
                xgb_n_est=st.session_state["xgb_n_est"],
                xgb_depth=st.session_state["xgb_depth"],
                xgb_lr=st.session_state["xgb_lr"],
                xgb_sub=st.session_state["xgb_sub"],
                xgb_col=st.session_state["xgb_col"],
                xgb_alpha=st.session_state["xgb_alpha"],
                xgb_lambda=st.session_state["xgb_lambda"],
                xgb_mcw=st.session_state["xgb_mcw"],
                lgb_n_est=st.session_state["lgb_n_est"],
                lgb_depth=st.session_state["lgb_depth"],
                lgb_lr=st.session_state["lgb_lr"],
                svm_c=st.session_state["svm_c"],
                mlp_layers=st.session_state["mlp_layers"],
                lr_c=st.session_state["lr_c"],
            )
        with tabs[4]:
            render_model_comparison(
                n_cv_folds=st.session_state["n_cv_folds"],
                test_size=st.session_state["test_size"],
                xgb_n_est=st.session_state["xgb_n_est"],
                xgb_depth=st.session_state["xgb_depth"],
                xgb_lr=st.session_state["xgb_lr"],
                xgb_sub=st.session_state["xgb_sub"],
                xgb_col=st.session_state["xgb_col"],
                xgb_alpha=st.session_state["xgb_alpha"],
                xgb_lambda=st.session_state["xgb_lambda"],
                xgb_mcw=st.session_state["xgb_mcw"],
                lgb_n_est=st.session_state["lgb_n_est"],
                lgb_depth=st.session_state["lgb_depth"],
                lgb_lr=st.session_state["lgb_lr"],
                svm_c=st.session_state["svm_c"],
                mlp_layers=st.session_state["mlp_layers"],
                lr_c=st.session_state["lr_c"],
            )

if __name__ == "__main__":
    main()