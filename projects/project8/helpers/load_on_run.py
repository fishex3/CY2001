import streamlit as st
import pandas as pd
import numpy as np
from .feature_helpers import fetch_prices, build_features, build_tlt_corr_features, get_tvp_var_spillover


def prepare_quarterly_features(feats, quarterly_index, lag_features=True):
    """
    Prepare quarterly features from daily feature DataFrames.
    
    Args:
        feats: Dict of feature DataFrames (daily indexed)
        quarterly_index: Quarterly DatetimeIndex
        lag_features: If True, shift features by one quarter (forecast mode).
                     If False, use contemporaneous features (nowcast mode).
    
    Returns:
        Dict of quarterly feature DataFrames, indexed by quarter end dates.
    """
    quarterly_feats = {}
    
    for feat_name, feat_df in feats.items():
        if isinstance(feat_df, pd.DataFrame):
            # Resample to quarterly (quarter-end)
            quarterly_feat = feat_df.resample('QE').last()
        elif isinstance(feat_df, pd.Series):
            quarterly_feat = feat_df.resample('QE').last()
        else:
            continue
            
        # Apply lag if requested
        if lag_features:
            quarterly_feat = quarterly_feat.shift(1)
            
        quarterly_feats[feat_name] = quarterly_feat
    
    return quarterly_feats


def load_on_run(
        selected_tickers,
        price_start,
        study_start,
        study_end,
        vol_window,
        var_conf,
        forecast_h,
        kappa1,
        kappa2,
        lag_features=True
):
    # Safe retrieval of the benchmark from session state
    BENCHMARK = st.session_state.get("BENCHMARK", "^GSPC")

    if True:
        if not selected_tickers:
            st.warning("Please select at least one sector ticker.")
            st.stop()

        with st.spinner("⏳ Downloading price data from yfinance…"):
            # TLT: same date range as other tickers (price_start → study_end); used for bond correlation features
            all_tickers = list(dict.fromkeys(selected_tickers + [BENCHMARK, "TLT"]))
            prices_all, volume_all = fetch_prices(all_tickers, price_start, study_end)
            if "TLT" not in prices_all.columns:
                st.error("Failed to download TLT (iShares 20+ Year Treasury ETF). Check the ticker and date range.")
                st.stop()

        sector_px = prices_all[[t for t in selected_tickers if t in prices_all.columns]]
        bm_px = prices_all[BENCHMARK]
        sector_vol = volume_all[[t for t in selected_tickers if t in volume_all.columns]]

        # Filter study window
        mask = (sector_px.index >= pd.Timestamp(study_start)) & (sector_px.index <= pd.Timestamp(study_end))
        sector_px_study = sector_px[mask]
        bm_px_study = bm_px[mask]
        sector_vol_study = sector_vol[mask]
        tlt_study = prices_all["TLT"].reindex(sector_px_study.index).ffill()

        with st.spinner("⏳ Computing features…"):
            feats = build_features(sector_px_study, bm_px_study, sector_vol_study, vol_window, var_conf)
            # Rolling correlation vs TLT
            feats.update(build_tlt_corr_features(sector_px_study, tlt_study, window=vol_window))

        with st.spinner("⏳ Computing TVP-VAR spillover(~5min)…"):
            result = get_tvp_var_spillover(
                sector_px_study,
                selected_tickers,
                vol_window,
                forecast_h,
                kappa1,
                kappa2
            )

        net_q = result["spillover_targets"].pivot(
            index="date",
            columns="sector",
            values="net_spillover"
        )
        tci_df = result["tci_quarterly"]

        # ---------------------------------------------------------
        # PANEL BUILDING LOGIC
        # ---------------------------------------------------------
        quarterly_index = sector_px_study.resample("QE").last().index

        # Prepare quarterly features (optionally lagged)
        quarterly_feats = prepare_quarterly_features(feats, quarterly_index, lag_features)

        # Normalize the R-generated dates to Pandas Quarters to ensure a perfect match
        net_q.index = pd.to_datetime(net_q.index).to_period("Q")

        panel_rows = []
        for qt in quarterly_index:
            current_q = qt.to_period("Q")
            
            # Skip if this quarter would have NaN features due to lagging
            if lag_features and qt == quarterly_index[0]:
                continue
                
            for t in selected_tickers:
                # 1. Initialize the row
                row = {"date": qt, "sector": t}

                # 2. Assign features from quarterly data
                for feat_name, quarterly_df in quarterly_feats.items():
                    try:
                        if isinstance(quarterly_df, pd.DataFrame) and t in quarterly_df.columns:
                            val = quarterly_df.loc[qt, t]
                            row[feat_name] = val if not pd.isna(val) else np.nan
                        elif isinstance(quarterly_df, pd.Series):
                            val = quarterly_df.loc[qt]
                            row[feat_name] = val if not pd.isna(val) else np.nan
                        else:
                            row[feat_name] = np.nan
                    except (KeyError, IndexError):
                        row[feat_name] = np.nan

                # 3. Assign spillover targets using the Period index
                if current_q in net_q.index and t in net_q.columns:
                    row["net_spillover"] = net_q.loc[current_q, t]
                    row["is_transmitter"] = int(row["net_spillover"] > 0)
                else:
                    # Provide defaults so the columns always exist
                    row["net_spillover"] = np.nan
                    row["is_transmitter"] = 0

                # 4. CRITICAL: Append only once per ticker, per quarter
                panel_rows.append(row)

        panel_df = pd.DataFrame(panel_rows)
        panel_df = panel_df.sort_values(["date", "sector"]).reset_index(drop=True)

        st.session_state.update({
            "data_loaded": True,
            "sector_px": sector_px_study,
            "bm_px": bm_px_study,
            "sector_vol": sector_vol_study,
            "calculated_features": feats,
            "quarterly_features": quarterly_feats,
            "net_q": net_q,
            "tci_df": tci_df,
            "panel_df": panel_df,
            "selected_tickers": selected_tickers,
            "vol_window": vol_window,
            "var_conf": var_conf,
            "lag_features": lag_features,
        })

    # Safe check in case data_loaded isn't in state yet
    if not st.session_state.get("data_loaded", False):
        st.info("👈 Configure parameters in the sidebar and click **Run Full Pipeline** to begin.")
        st.stop()