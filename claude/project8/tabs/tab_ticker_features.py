import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from helpers.feature_helpers import compute_log_returns

def render_ticker_features(
        sector_px,
        tickers,
):


    SECTORS = st.session_state["SECTORS"]
    PALETTE = st.session_state["PALETTE"]
    st.markdown('<div class="section-header">Adjusted Close Prices (Indexed to 100)</div>', unsafe_allow_html=True)
    indexed = (sector_px / sector_px.iloc[0]) * 100
    fig = go.Figure()
    for i, t in enumerate(tickers):
        fig.add_trace(go.Scatter(x=indexed.index, y=indexed[t], name=f"{t} – {SECTORS.get(t,'')}",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=1.8)))
    fig.update_layout(height=380, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Daily Log Returns</div>', unsafe_allow_html=True)
        lr = compute_log_returns(sector_px).dropna()
        fig2 = go.Figure()
        for i, t in enumerate(tickers):
            fig2.add_trace(go.Scatter(x=lr.index, y=lr[t], name=t,
                                      line=dict(width=0.8, color=PALETTE[i % len(PALETTE)]),
                                      opacity=0.7))
        fig2.update_layout(height=300, hovermode="x unified",
                           legend=dict(orientation="h", y=-0.25),
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Return Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = lr.corr()
        fig3 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         text_auto=".2f", aspect="auto")
        fig3.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, width="stretch")

    st.markdown('<div class="section-header">Return Distribution</div>', unsafe_allow_html=True)
    ticker_sel = st.selectbox("Select ticker for distribution", tickers, key="ret_dist")
    fig4 = px.histogram(lr[ticker_sel].dropna(), nbins=80, histnorm="probability density",
                        color_discrete_sequence=["#2E5FA3"])
    fig4.update_layout(height=280, xaxis_title="Log Return", yaxis_title="Density",
                       margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig4, width="stretch")