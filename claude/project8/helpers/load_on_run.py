import streamlit as st
import pandas as pd
import numpy as np
from .feature_helpers import fetch_prices, build_features, get_tvp_var_spillover


def load_on_run(
        selected_tickers,
        price_start,
        study_start,
        study_end,
        vol_window,
        var_conf,
        forecast_h,
        kappa1,
        kappa2
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

        with st.spinner("⏳ Computing features…"):
            feats = build_features(sector_px_study, bm_px_study, sector_vol_study, vol_window, var_conf)

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

        # Normalize the R-generated dates to Pandas Quarters to ensure a perfect match
        net_q.index = pd.to_datetime(net_q.index).to_period("Q")

        panel_rows = []
        for qt in quarterly_index:
            current_q = qt.to_period("Q")
            window_data = sector_px_study[sector_px_study.index <= qt]

            if window_data.empty:
                continue

            for t in selected_tickers:
                # 1. Initialize the row
                row = {"date": qt, "sector": t}

                # 2. Assign features cleanly
                for feat_name, feat_df in feats.items():
                    try:
                        if isinstance(feat_df, pd.DataFrame) and t in feat_df.columns:
                            val = feat_df.loc[:qt, t].dropna()
                            row[feat_name] = val.iloc[-1] if not val.empty else np.nan
                        elif isinstance(feat_df, pd.Series):
                            val = feat_df.loc[:qt].dropna()
                            row[feat_name] = val.iloc[-1] if not val.empty else np.nan
                        else:
                            row[feat_name] = np.nan
                    except:
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
            "net_q": net_q,
            "tci_df": tci_df,
            "panel_df": panel_df,
            "selected_tickers": selected_tickers,
            "vol_window": vol_window,
            "var_conf": var_conf,
        })

    # Safe check in case data_loaded isn't in state yet
    if not st.session_state.get("data_loaded", False):
        st.info("👈 Configure parameters in the sidebar and click **Run Full Pipeline** to begin.")
        st.stop()