import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np


def render_spillover_analysis(
        tci_df,
        net_q,
        tickers,
        panel_df,
):
    # Safely get session state variables
    PALETTE = st.session_state["PALETTE"]
    SECTORS = st.session_state["SECTORS"]
    st.markdown('<div class="section-header">Total Connectedness Index (TCI)</div>', unsafe_allow_html=True)
    if not tci_df.empty:
        fig_tci = go.Figure()
        fig_tci.add_trace(go.Scatter(x=tci_df["date"], y=tci_df["tci"],
                                     fill="tozeroy", fillcolor="rgba(46,95,163,0.15)",
                                     line=dict(color="#2E5FA3", width=2), name="TCI"))
        fig_tci.update_layout(height=300, yaxis_title="TCI (%)",
                               hovermode="x unified", margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_tci, width="stretch")

    st.markdown('<div class="section-header">NET Spillover Heatmap (Quarterly)</div>', unsafe_allow_html=True)
    if not net_q.empty:
        net_plot = net_q[[t for t in tickers if t in net_q.columns]].dropna(how="all")
        if not net_plot.empty:
            # Plotly/Streamlit can't JSON-serialize pandas Period objects (e.g. PeriodIndex).
            # Convert the quarterly index to strings so it is safe for Plotly.
            net_plot_safe = net_plot.copy()
            if isinstance(net_plot_safe.index, pd.PeriodIndex):
                net_plot_safe.index = net_plot_safe.index.astype(str)
            else:
                # handle Period values stored in a generic Index
                net_plot_safe.index = net_plot_safe.index.map(lambda x: str(x))
            fig_heat = px.imshow(
                net_plot_safe.T,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                labels=dict(x="Quarter", y="Sector", color="NET"),
                aspect="auto",
            )
            fig_heat.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_heat, width="stretch")

    st.markdown('<div class="section-header">NET Spillover Over Time — Per Sector</div>', unsafe_allow_html=True)
    fig_net = go.Figure()
    for i, t in enumerate(tickers):
        if t in net_q.columns:
            series = net_q[t].dropna()
            x = series.index
            if isinstance(x, pd.PeriodIndex):
                x = x.to_timestamp()
            fig_net.add_trace(go.Scatter(x=x, y=series, name=t,
                                          line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig_net.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig_net.update_layout(height=350, hovermode="x unified",
                           legend=dict(orientation="h", y=-0.2),
                           margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_title="NET Spillover")
    st.plotly_chart(fig_net, width="stretch")

    st.markdown('<div class="section-header">Transmitter / Receiver Classification</div>', unsafe_allow_html=True)
    if "is_transmitter" in panel_df.columns:
        trans_rate = panel_df.groupby("sector")["is_transmitter"].mean().reset_index()
        trans_rate.columns = ["sector", "transmitter_rate"]
        trans_rate["label"] = trans_rate["sector"].map(SECTORS)
        fig_tr = px.bar(trans_rate, x="sector", y="transmitter_rate",
                        color="transmitter_rate", color_continuous_scale="RdBu",
                        color_continuous_midpoint=0.5,
                        labels={"transmitter_rate": "Transmitter Rate", "sector": "Sector"},
                        text_auto=".1%")
        fig_tr.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig_tr.update_layout(height=320, coloraxis_showscale=False,
                              margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_tr, width="stretch")
