import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from tvp_var_spillover import get_tvp_var_spillover
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve
)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, start, end):
    raw = yf.download(tickers, start=str(start), end=str(end),
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
        volume = raw["Volume"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    prices = prices.ffill().dropna(how="all")
    volume = volume.ffill().dropna(how="all")
    return prices, volume


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1))


def compute_volatility(prices, window):
    lr = compute_log_returns(prices)
    return lr.rolling(window).std() * np.sqrt(252)


def compute_downside_vol(prices, window):
    lr = compute_log_returns(prices)
    neg = lr.where(lr < 0, 0)
    return neg.rolling(window).std() * np.sqrt(252)


def compute_beta(prices, bm, window):
    lr = compute_log_returns(prices)
    lrbm = compute_log_returns(bm)
    out = pd.DataFrame(index=prices.index)
    for c in prices.columns:
        cov = lr[c].rolling(window).cov(lrbm)
        var = lrbm.rolling(window).var()
        out[c] = cov / var
    return out


def compute_market_corr(prices, bm, window):
    lr = compute_log_returns(prices)
    lrbm = compute_log_returns(bm)
    out = pd.DataFrame(index=prices.index)
    for c in prices.columns:
        out[c] = lr[c].rolling(window).corr(lrbm)
    return out


def _expand_lagged_quarterly_to_daily(quarterly_df: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Map each trading day to the lagged quarter-end feature for its calendar quarter.

    quarterly_df must be indexed at quarter-end dates (e.g. resample('QE').last()),
    then .shift(1) so the row labeled quarter Q holds data that only uses information
    through Q-1 (no look-ahead). After that shift, every day in quarter Q should use
    the value stored on Q's row (constant within the quarter for modeling).
    """
    ql = quarterly_df.copy()
    ql.index = ql.index.to_period("Q")
    per = pd.DatetimeIndex(daily_index).to_period("Q")
    out = ql.reindex(per)
    out.index = daily_index
    return out


def build_tlt_corr_features(sector_px: pd.DataFrame, tlt_px: pd.Series, window: int = 30) -> dict[str, pd.DataFrame]:
    """
    Sector vs TLT (bonds) rolling correlation, aggregated to quarters.

    Returns contemporaneous quarterly features. Lagging is handled separately.
    """
    tlt = tlt_px.reindex(sector_px.index).ffill()
    lr_sec = compute_log_returns(sector_px)
    lr_tlt = np.log(tlt / tlt.shift(1))

    # Daily rolling correlation vs TLT
    corr_daily = lr_sec.rolling(window, min_periods=window).corr(lr_tlt)
    corr_daily.columns = [f"{c}_corr_tlt" for c in sector_px.columns]

    # Quarter-end snapshot of the rolling correlation series
    quarterly_corr = corr_daily.resample("QE").last()
    corr_tlt = _expand_lagged_quarterly_to_daily(quarterly_corr, sector_px.index)
    corr_tlt.columns = list(sector_px.columns)

    # QoQ change in quarter-end correlation
    quarterly_diff = quarterly_corr.diff()
    corr_tlt_diff = _expand_lagged_quarterly_to_daily(quarterly_diff, sector_px.index)
    corr_tlt_diff.columns = list(sector_px.columns)

    return {"corr_tlt": corr_tlt, "corr_tlt_diff": corr_tlt_diff}


def compute_amihud(prices, volume):
    lr = compute_log_returns(prices)
    dvol = prices * volume
    out = pd.DataFrame(index=prices.index)
    for c in prices.columns:
        daily = (lr[c].abs() / dvol[c].replace(0, np.nan)) * 1e6
        monthly = daily.resample("ME").mean()
        out[c] = monthly.reindex(prices.index, method="ffill")
    return out


def compute_var(prices, window, conf):
    lr = compute_log_returns(prices)
    out = pd.DataFrame(index=prices.index)
    for c in prices.columns:
        out[c] = lr[c].rolling(window).quantile(1 - conf / 100)
    return out


def compute_skewness(prices, window):
    lr = compute_log_returns(prices)
    return lr.rolling(window).skew()


def compute_kurtosis(prices, window):
    lr = compute_log_returns(prices)
    return lr.rolling(window).kurt()


def compute_avg_sector_corr(prices, window):
    lr = compute_log_returns(prices)
    series = pd.Series(index=prices.index, dtype=float)
    tickers = prices.columns.tolist()
    corr_matrix = lr.rolling(window).corr()
    for dt in prices.index:
        try:
            cm = corr_matrix.loc[dt]
            vals = []
            for i, t1 in enumerate(tickers):
                for j, t2 in enumerate(tickers):
                    if i < j:
                        try:
                            vals.append(cm.loc[t1, t2])
                        except:
                            pass
            series[dt] = np.nanmean(vals) if vals else np.nan
        except:
            series[dt] = np.nan
    return series


def build_features(sector_px, bm_px, sector_vol, window, var_conf):
    ETF_MARKET_CAP = st.session_state["ETF_MARKET_CAP"]
    feats = {"volatility": compute_volatility(sector_px, window),
             "downside_volatility": compute_downside_vol(sector_px, window),
             "beta": compute_beta(sector_px, bm_px, window),
             "market_correlation": compute_market_corr(sector_px, bm_px, window),
             "amihud_illiquidity": compute_amihud(sector_px, sector_vol), "volume": np.log(sector_vol + 1),
             "var": compute_var(sector_px, window, var_conf), "skewness": compute_skewness(sector_px, window),
             "kurtosis": compute_kurtosis(sector_px, window), "log_market_cap": pd.DataFrame(
            {t: np.log(ETF_MARKET_CAP.get(t, 10e9)) for t in sector_px.columns},
            index=sector_px.index
        ), "average_sector_correlation": compute_avg_sector_corr(sector_px, window)}
    return feats

def compute_tvp_var_spillover(prices_raw, sectors, vol_window, forecast_h, kappa1, kappa2):
    return get_tvp_var_spillover(
        prices_raw,
        sectors,
        vol_window,
        forecast_h,
        kappa1,
        kappa2
    )

def get_models(xgb_p, lgb_p, svm_c, mlp_hidden, lr_c, rs=42):
    try:
        hidden = tuple(int(x.strip()) for x in mlp_hidden.strip("()").split(","))
    except:
        hidden = (32, 16)
    return {
        "XGBoost":  xgb.XGBClassifier(**xgb_p),
        "LightGBM": lgb.LGBMClassifier(**lgb_p),
        "SVM":      SVC(kernel="rbf", C=svm_c, probability=True, random_state=rs),
        "MLP":      MLPClassifier(hidden_layer_sizes=hidden, alpha=0.01, max_iter=500, random_state=rs),
        "LR":       LogisticRegression(C=lr_c, max_iter=1000, random_state=rs),
    }

def compute_metrics(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    try:
        yprob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, yprob)
    except:
        auc = np.nan
    return {
        "accuracy":  round(accuracy_score(y_te, yp), 4),
        "auc":       round(auc, 4) if not np.isnan(auc) else np.nan,
        "precision": round(precision_score(y_te, yp, zero_division=0), 4),
        "recall":    round(recall_score(y_te, yp, zero_division=0), 4),
        "f1":        round(f1_score(y_te, yp, zero_division=0), 4),
    }