import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve
)


from helpers.feature_helpers import get_models, compute_metrics


def render_model_comparison(
        n_cv_folds,
        test_size,
        xgb_n_est,
        xgb_depth,
        xgb_lr,
        xgb_sub,
        xgb_col,
        xgb_alpha,
        xgb_lambda,
        xgb_mcw,
        lgb_n_est,
        lgb_depth,
        lgb_lr,
        svm_c,
        mlp_layers,
        lr_c
):
    PALETTE = st.session_state["PALETTE"]
    X = st.session_state["X"]
    y = st.session_state["y"]
    st.markdown('<div class="section-header">In-Sample Model Comparison (All Models)</div>', unsafe_allow_html=True)
    xgb_params = dict(
        n_estimators=xgb_n_est, max_depth=xgb_depth, learning_rate=xgb_lr,
        subsample=xgb_sub, colsample_bytree=xgb_col,
        reg_alpha=xgb_alpha, reg_lambda=xgb_lambda,
        min_child_weight=xgb_mcw, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    lgb_params = dict(
        n_estimators=lgb_n_est, max_depth=lgb_depth, learning_rate=lgb_lr,
        subsample=xgb_sub, colsample_bytree=xgb_col,
        random_state=42, verbose=-1,
    )
    if len(X) < 10:
        st.warning("Not enough data for model comparison.")
        st.stop()

    models_dict = get_models(xgb_params, lgb_params, svm_c, mlp_layers, lr_c)
    rows = []

    with st.spinner("Training all 5 models…"):
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        for name, mdl in models_dict.items():
            try:
                m = compute_metrics(mdl, X_sc, y, X_sc, y)
                m["model"] = name
                rows.append(m)
            except Exception as e:
                rows.append({"model": name, "accuracy": np.nan, "auc": np.nan,
                             "precision": np.nan, "recall": np.nan, "f1": np.nan})

    insample_df = pd.DataFrame(rows).set_index("model")

    # Bar comparison
    metrics_bar = ["accuracy", "auc", "precision", "recall", "f1"]
    colors_bar = ["#2E5FA3", "#3B8BD4", "#888780", "#BA7517", "#D85A30"]
    fig_comp = make_subplots(rows=1, cols=5, subplot_titles=[m.upper() for m in metrics_bar])
    for idx, (metric, color) in enumerate(zip(metrics_bar, colors_bar)):
        vals = insample_df[metric].values
        mods = insample_df.index.tolist()
        fig_comp.add_trace(go.Bar(x=mods, y=vals, marker_color=color,
                                  name=metric, showlegend=False,
                                  text=[f"{v:.2f}" for v in vals],
                                  textposition="outside"), row=1, col=idx + 1)
        fig_comp.update_yaxes(range=[0, 1.1], row=1, col=idx + 1)

    fig_comp.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_comp, width="stretch")

    st.markdown('<div class="section-header">Results Table</div>', unsafe_allow_html=True)
    st.dataframe(
        insample_df.style.format("{:.4f}"),
        width="stretch"
    )

    # Radar chart
    st.markdown('<div class="section-header">Model Radar Chart</div>', unsafe_allow_html=True)
    fig_radar = go.Figure()
    for i, model_name in enumerate(insample_df.index):
        vals = insample_df.loc[model_name, metrics_bar].values.tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metrics_bar + [metrics_bar[0]],
            name=model_name,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fill="toself", fillcolor=PALETTE[i % len(PALETTE)],
            opacity=0.15,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=-0.1)
    )
    st.plotly_chart(fig_radar, width="stretch")

    # ROC curves for all models
    st.markdown('<div class="section-header">ROC Curves — All Models</div>', unsafe_allow_html=True)
    fig_rocs = go.Figure()
    for i, (name, mdl) in enumerate(models_dict.items()):
        try:
            sc2 = StandardScaler()
            Xs2 = sc2.fit_transform(X)
            mdl2 = type(mdl)(**mdl.get_params())
            mdl2.fit(Xs2, y)
            yp2 = mdl2.predict_proba(Xs2)[:, 1]
            fpr2, tpr2, _ = roc_curve(y, yp2)
            auc2 = roc_auc_score(y, yp2)
            fig_rocs.add_trace(go.Scatter(
                x=fpr2, y=tpr2, name=f"{name} (AUC={auc2:.3f})",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2)
            ))
        except:
            pass

    fig_rocs.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Random"))
    fig_rocs.update_layout(height=420, xaxis_title="FPR", yaxis_title="TPR",
                           legend=dict(orientation="h", y=-0.2),
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_rocs, width="stretch")
