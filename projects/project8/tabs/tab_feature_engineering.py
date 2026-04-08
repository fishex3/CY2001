import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def render_feature_engineering(
        calculated_features,
        tickers,
        selected_features,
        panel_df
):

    PALETTE = st.session_state["PALETTE"]
    feat_select = st.selectbox("Feature to display", selected_features, key="selected_features_feature_engineering")
    feat_data = calculated_features.get(feat_select)

    if isinstance(feat_data, pd.DataFrame):
        st.markdown(f'<div class="section-header">{feat_select.replace("_"," ").title()} — All Sectors</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for i, t in enumerate(tickers):
            if t in feat_data.columns:
                fig.add_trace(go.Scatter(x=feat_data.index, y=feat_data[t], name=t,
                                         line=dict(color=PALETTE[i % len(PALETTE)], width=1.5)))
        fig.update_layout(height=360, hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="section-header">Quarterly Cross-Sector Distribution</div>', unsafe_allow_html=True)
        q_mean = feat_data.resample("QE").mean()
        fig_box = go.Figure()
        for i, t in enumerate(tickers):
            if t in q_mean.columns:
                fig_box.add_trace(go.Box(y=q_mean[t].dropna(), name=t,
                                         marker_color=PALETTE[i % len(PALETTE)]))
        fig_box.update_layout(height=300, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_box, width="stretch")

    elif isinstance(feat_data, pd.Series):
        st.markdown(f'<div class="section-header">{feat_select.replace("_"," ").title()}</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(x=feat_data.index, y=feat_data, line=dict(color="#2E5FA3", width=2)))
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, width="stretch")

    # Feature correlation matrix at quarterly level
    st.markdown('<div class="section-header">Feature Correlation Matrix (Quarterly Panel)</div>', unsafe_allow_html=True)
    feat_cols = [c for c in selected_features if c in panel_df.columns]
    if feat_cols:
        corr_f = panel_df[feat_cols].dropna().corr()
        fig_fc = px.imshow(corr_f, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           text_auto=".2f", aspect="auto")
        fig_fc.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_fc, width="stretch")