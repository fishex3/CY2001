import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve
)

import xgboost as xgb
import shap

def render_ml_pipeline(
        selected_features,
        panel_df,
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
    feat_cols = [c for c in selected_features if c in panel_df.columns]

    if "is_transmitter" not in panel_df.columns or len(feat_cols) == 0:
        st.warning("Insufficient data for ML pipeline. Need spillover targets and at least one feature.")
        st.stop()

    valid_mask = panel_df[feat_cols + ["is_transmitter"]].notna().all(axis=1)
    df_ml = panel_df[valid_mask].copy()
    X = df_ml[feat_cols].copy()
    y = df_ml["is_transmitter"].astype(int)

    # Impute remaining
    X = X.ffill().bfill().fillna(0)

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", len(X))
    c2.metric("Features Used", len(feat_cols))
    c3.metric("Transmitters", int(y.sum()))
    c4.metric("Transmitter Rate", f"{y.mean():.1%}")

    # XGBoost params
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

    if len(X) < 20:
        st.warning("Not enough data for ML. Try a wider date range.")
        st.stop()

    # ── Feature Importance ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">XGBoost Feature Importance</div>', unsafe_allow_html=True)

    with st.spinner("Training XGBoost for feature importance…"):
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feat_cols)
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(X_scaled, y)

        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "importance": model_xgb.feature_importances_,
        }).sort_values("importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=imp_df["importance"], y=imp_df["feature"],
        orientation="h", marker_color="#2E5FA3",
    ))
    fig_imp.update_layout(height=max(300, len(feat_cols) * 30),
                          xaxis_title="Importance", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_imp, width="stretch")

    # ── SHAP ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">SHAP Mean |Value| by Feature</div>', unsafe_allow_html=True)
    with st.spinner("Computing SHAP values…"):
        explainer = shap.TreeExplainer(model_xgb)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_mean = np.abs(shap_vals).mean(axis=0)
        shap_dir = shap_vals.mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feat_cols,
            "shap_importance": shap_mean,
            "direction": shap_dir,
        }).sort_values("shap_importance", ascending=True)

    colors = ["#A32D2D" if d > 0 else "#185FA5" for d in shap_df["direction"]]
    fig_shap = go.Figure(go.Bar(
        x=shap_df["shap_importance"], y=shap_df["feature"],
        orientation="h", marker_color=colors,
    ))
    fig_shap.update_layout(height=max(300, len(feat_cols) * 30),
                           xaxis_title="Mean |SHAP|",
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_shap, width="stretch")
    st.caption("🔴 Red = positive impact on transmitter prediction   🔵 Blue = negative impact")

    # ── Confusion Matrix & ROC ──────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Confusion Matrix (In-Sample)</div>', unsafe_allow_html=True)
        y_pred = model_xgb.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual"),
                           x=["Receiver", "Transmitter"],
                           y=["Receiver", "Transmitter"])
        fig_cm.update_layout(height=300, coloraxis_showscale=False,
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cm, width="stretch")

    with col2:
        st.markdown('<div class="section-header">ROC Curve (In-Sample)</div>', unsafe_allow_html=True)
        y_prob = model_xgb.predict_proba(X_scaled)[:, 1]
        try:
            fpr, tpr, _ = roc_curve(y, y_prob)
            auc_score = roc_auc_score(y, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"XGBoost (AUC={auc_score:.3f})",
                                         line=dict(color="#2E5FA3", width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(color="gray", dash="dash"), name="Random"))
            fig_roc.update_layout(height=300,
                                  xaxis_title="FPR", yaxis_title="TPR",
                                  margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_roc, width="stretch")
        except:
            st.info("Insufficient class diversity for ROC curve.")

    # ── Time Series CV ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Time Series Cross-Validation</div>', unsafe_allow_html=True)
    with st.spinner("Running time series CV…"):
        dates = np.sort(df_ml["date"].unique())
        fold_size = max(test_size, len(dates) // (n_cv_folds + 1))
        cv_results = []

        for fold in range(n_cv_folds):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = min(test_start + test_size, len(dates))
            if test_end >= len(dates):
                break

            train_dates = dates[:train_end]
            test_dates = dates[test_start:test_end]
            tr_mask = df_ml["date"].isin(train_dates)
            te_mask = df_ml["date"].isin(test_dates)

            X_tr, y_tr = X[tr_mask].values, y[tr_mask].values
            X_te, y_te = X[te_mask].values, y[te_mask].values

            if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
                continue

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            m = xgb.XGBClassifier(**xgb_params)
            m.fit(X_tr_s, y_tr)
            yp = m.predict(X_te_s)
            try:
                yprob = m.predict_proba(X_te_s)[:, 1]
                auc = roc_auc_score(y_te, yprob)
            except:
                auc = np.nan

            cv_results.append({
                "fold": fold + 1,
                "train_end": pd.Timestamp(train_dates[-1]).date(),
                "test_end": pd.Timestamp(test_dates[-1]).date(),
                "accuracy": accuracy_score(y_te, yp),
                "auc": auc,
                "precision": precision_score(y_te, yp, zero_division=0),
                "recall": recall_score(y_te, yp, zero_division=0),
                "f1": f1_score(y_te, yp, zero_division=0),
            })

    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        fig_cv = make_subplots(rows=2, cols=2, subplot_titles=["Accuracy", "AUC", "Precision", "Recall"])
        metrics_cv = [("accuracy", "Accuracy", 1, 1), ("auc", "AUC", 1, 2),
                      ("precision", "Precision", 2, 1), ("recall", "Recall", 2, 2)]
        for col_name, title, r, c in metrics_cv:
            vals = cv_df[col_name].dropna()
            fig_cv.add_trace(go.Scatter(x=cv_df["fold"][:len(vals)], y=vals,
                                        mode="lines+markers", name=title,
                                        line=dict(color="#2E5FA3", width=2),
                                        marker=dict(size=8)), row=r, col=c)
            fig_cv.add_hline(y=vals.mean(), line_dash="dash", line_color="red",
                             opacity=0.6, row=r, col=c)
            fig_cv.update_yaxes(range=[0, 1], row=r, col=c)

        fig_cv.update_layout(height=450, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_cv, width="stretch")

        st.markdown('<div class="section-header">CV Summary Stats</div>', unsafe_allow_html=True)
        summary = cv_df[["accuracy", "auc", "precision", "recall", "f1"]].agg(["mean", "std"])
        st.dataframe(summary.style.format("{:.3f}"), width="stretch")
    else:
        st.warning("Not enough data for cross-validation folds. Try a wider date range.")
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.markdown('<div class="section-header">In-Sample Summary Stats</div>', unsafe_allow_html=True)
    
    in_sample_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score"],
        "Value": [
            accuracy_score(y, y_pred),
            auc_score, 
            precision_score(y, y_pred, zero_division=0),
            recall_score(y, y_pred, zero_division=0),
            f1_score(y, y_pred, zero_division=0)
        ]
    })
    
    st.dataframe(in_sample_metrics.style.format({"Value": "{:.3f}"}), hide_index=True, width="stretch")