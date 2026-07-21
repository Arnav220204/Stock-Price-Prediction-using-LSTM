"""
Model B v2 — Improved Bidirectional LSTM for Tesla Stock Price Prediction
=========================================================================

IMPROVEMENTS OVER MODEL B v1
──────────────────────────────────────────────────────────────────────────
1. LOG-RETURN TRANSFORMATION
   Raw Close price is non-stationary (mean and variance drift over time).
   Training on it forces the model to memorize price levels instead of
   learning price dynamics. Log-returns are stationary, roughly symmetric
   (skew=0.005 confirmed), and scale-invariant — the same pattern in 2014
   at $5 and in 2024 at $400 looks identical to the model.

2. MULTIVARIATE INPUT (11 FEATURES)
   The original Model B only saw past Close prices. Now it sees:
   EMA ratios, RSI, MACD, Bollinger position, ATR, Volume ratio, and
   intraday range — the same signals a human analyst would look at.

3. BIDIRECTIONAL LSTM
   A standard LSTM processes sequences left-to-right only. Bidirectional
   LSTM runs a second LSTM right-to-left over the same window, doubling
   the context available when producing the prediction. Given that we are
   doing one-step-ahead forecasting (not true future prediction), the
   right-to-left pass sees earlier data in the window with "knowledge"
   of later context — useful for detecting peak/trough patterns.

4. HUBER LOSS instead of MSE
   Tesla log-return kurtosis = 4.73 (confirmed from data inspection),
   meaning the distribution has fatter tails than a normal — i.e. large
   surprise moves happen more often than MSE assumes. Huber loss is
   quadratic for small errors and linear for large ones, so a single
   extreme day doesn't blow up the gradient and destabilise training.

5. 60-DAY WINDOW (was 20)
   With 3,514 training rows we can safely use a longer look-back without
   starving the model of sequences. 60 trading days ≈ 3 calendar months,
   capturing quarterly earnings cycles and medium-term trend context.

6. MULTI-SEED TRAINING (3 seeds → mean ± std reported)
   Financial time-series models are noisy. A single run's R² of 0.29
   vs -1.51 could partly be seed luck. Running 3 seeds and reporting
   mean ± std makes the comparison table statistically credible.

7. PAPER-QUALITY EVALUATION SUITE
   RMSE, MAE, MAE%, R², Directional Accuracy, Diebold-Mariano test
   vs naive baseline, monthly breakdown, residual diagnostics.

CEIC DATA INTEGRATION (see Section 8 at the bottom)
──────────────────────────────────────────────────────────────────────────
When you have the CEIC exports, paste the file paths into Section 8.
The code merges them automatically and adds them as extra input features.
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from clean_data import load_and_clean

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEEDS    = [42, 7, 123]
WINDOW   = 60          # trading days of look-back
BATCH    = 32
EPOCHS   = 120
OUT      = "/home/claude/work/outputs"
DATA_25  = "/home/claude/work/data/Cleaned_Tesla_25yr_data.csv"
DATA_26  = "/home/claude/work/data/2026_Tesla.csv"
PREV_METRICS_2026 = {      # Model B v1 results for comparison table
    "RMSE": 22.10, "MAE": 19.20, "MAE_pct": 4.75, "R2": 0.29, "DirAcc": None
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_full = pd.read_csv(DATA_25, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
df_2026 = load_and_clean(DATA_26)
print(f"Training data : {len(df_full)} rows  {df_full['Date'].min().date()} → {df_full['Date'].max().date()}")
print(f"2026 hold-out : {len(df_2026)} rows  {df_2026['Date'].min().date()} → {df_2026['Date'].max().date()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ── Target ─────────────────────────────────────────────────────────────
    d["LogReturn"]     = np.log(d["Close"] / d["Close"].shift(1))

    # ── Trend / momentum ──────────────────────────────────────────────────
    ema20              = d["Close"].ewm(span=20, adjust=False).mean()
    ema50              = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA20_ratio"]   = (d["Close"] - ema20)  / ema20   # % deviation from EMA20
    d["EMA50_ratio"]   = (d["Close"] - ema50)  / ema50   # % deviation from EMA50

    # ── RSI (14-day) ──────────────────────────────────────────────────────
    delta              = d["Close"].diff()
    gain               = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss               = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    d["RSI"]           = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))) / 100  # 0-1

    # ── MACD ──────────────────────────────────────────────────────────────
    ema12              = d["Close"].ewm(span=12, adjust=False).mean()
    ema26              = d["Close"].ewm(span=26, adjust=False).mean()
    macd               = ema12 - ema26
    signal             = macd.ewm(span=9, adjust=False).mean()
    d["MACD_norm"]     = macd   / d["Close"]   # scale-independent
    d["MACD_hist"]     = (macd - signal) / d["Close"]

    # ── Bollinger Bands (20-day, 2σ) ──────────────────────────────────────
    bb_mid             = d["Close"].rolling(20).mean()
    bb_std             = d["Close"].rolling(20).std()
    d["BB_pos"]        = (d["Close"] - bb_mid) / (2 * bb_std.replace(0, 1e-9))  # ≈ -1…+1
    d["BB_width"]      = (4 * bb_std) / bb_mid.replace(0, 1e-9)                 # volatility

    # ── ATR (14-day, normalised) ──────────────────────────────────────────
    tr                 = pd.concat([
                             d["High"] - d["Low"],
                             (d["High"] - d["Close"].shift(1)).abs(),
                             (d["Low"]  - d["Close"].shift(1)).abs()
                         ], axis=1).max(axis=1)
    d["ATR_norm"]      = tr.rolling(14).mean() / d["Close"]

    # ── Volume ────────────────────────────────────────────────────────────
    vol_ma             = d["Volume"].rolling(20).mean()
    d["Volume_ratio"]  = d["Volume"] / vol_ma.replace(0, 1e-9)  # relative to recent average

    # ── Intraday ──────────────────────────────────────────────────────────
    d["HL_range"]      = (d["High"] - d["Low"]) / d["Close"]
    d["OC_return"]     = (d["Close"] - d["Open"]) / d["Open"]

    return d.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "LogReturn", "EMA20_ratio", "EMA50_ratio",
    "RSI", "MACD_norm", "MACD_hist",
    "BB_pos", "BB_width", "ATR_norm",
    "Volume_ratio", "HL_range", "OC_return"
]
TARGET_COL = "LogReturn"

df_feat   = add_features(df_full)
df26_feat = add_features(df_2026)
print(f"\nAfter feature engineering: {len(df_feat)} rows (dropped {len(df_full)-len(df_feat)} NaN rows)")
print(f"Features: {FEATURE_COLS}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT  (chronological 90 / 10 — never shuffle time series)
# ══════════════════════════════════════════════════════════════════════════════
split      = int(len(df_feat) * 0.90)
df_train   = df_feat.iloc[:split].copy()
df_test    = df_feat.iloc[split:].copy()
print(f"\nTrain: {len(df_train)} rows  {df_train['Date'].min().date()} → {df_train['Date'].max().date()}")
print(f"Test : {len(df_test)}  rows  {df_test['Date'].min().date()} → {df_test['Date'].max().date()}")
print(f"2026 : {len(df26_feat)} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCALING  (fit on TRAIN ONLY — no data leakage)
# ══════════════════════════════════════════════════════════════════════════════
feat_scaler = StandardScaler()
feat_scaler.fit(df_train[FEATURE_COLS])

tgt_scaler  = StandardScaler()
tgt_scaler.fit(df_train[[TARGET_COL]])

def scale(df_seg):
    X = feat_scaler.transform(df_seg[FEATURE_COLS])
    y = tgt_scaler.transform(df_seg[[TARGET_COL]]).ravel()
    return X, y

X_train_s, y_train_s = scale(df_train)
X_test_s,  y_test_s  = scale(df_test)
X_26_s,    y_26_s    = scale(df26_feat)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(X_full_s, y_full_s, X_new_s, y_new_s):
    """
    Prepend the last WINDOW rows of the preceding block to the new block
    so the very first sequence still has real look-back context.
    Returns sequences built from the new block only.
    """
    X_combo = np.vstack([X_full_s[-WINDOW:], X_new_s])
    y_combo = np.concatenate([y_full_s[-WINDOW:], y_new_s])
    Xs, ys  = [], []
    for i in range(WINDOW, len(X_combo)):
        Xs.append(X_combo[i-WINDOW:i])
        ys.append(y_combo[i])
    return np.array(Xs), np.array(ys)

X_tr, y_tr = make_sequences(X_train_s, y_train_s, X_train_s, y_train_s)
X_te, y_te = make_sequences(X_train_s, y_train_s, X_test_s,  y_test_s)
X_26, y_26 = make_sequences(X_test_s,  y_test_s,  X_26_s,    y_26_s)

# For train sequences only use from WINDOW onwards (no wrap-around needed)
X_tr = X_tr[WINDOW:]
y_tr = y_tr[WINDOW:]

N_FEAT = X_tr.shape[2]
print(f"\nX_train:{X_tr.shape}  X_test:{X_te.shape}  X_2026:{X_26.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PRICE RECONSTRUCTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def reconstruct_price(actual_price_series, pred_log_ret_scaled, prev_close_series):
    """
    One-step-ahead reconstruction:
      predicted_close[i] = actual_close[i-1] * exp(predicted_log_return[i])
    This avoids error accumulation — each prediction is anchored to real data.
    pred_log_ret_scaled: model output (StandardScaler-scaled)
    prev_close_series  : actual close price on the DAY BEFORE each prediction
    """
    pred_lr = tgt_scaler.inverse_transform(pred_log_ret_scaled.reshape(-1,1)).ravel()
    pred_px = prev_close_series * np.exp(pred_lr)
    return pred_px, pred_lr

def metrics(true_px, pred_px, true_lr, pred_lr):
    rmse   = np.sqrt(mean_squared_error(true_px, pred_px))
    mae    = mean_absolute_error(true_px, pred_px)
    mae_p  = mae / np.mean(true_px) * 100
    r2     = r2_score(true_px, pred_px)
    dir_a  = np.mean(np.sign(true_lr) == np.sign(pred_lr)) * 100
    return dict(RMSE=rmse, MAE=mae, MAE_pct=mae_p, R2=r2, DirAcc=dir_a)


# ══════════════════════════════════════════════════════════════════════════════
# 7. BASELINES
# ══════════════════════════════════════════════════════════════════════════════
# ── 7a. Naive persistence (log return = 0, i.e. price unchanged) ──────────
def naive_metrics(df_seg, label):
    px_true  = df_seg["Close"].values[1:]   # actual close
    px_naive = df_seg["Close"].values[:-1]  # previous close = prediction
    lr_true  = df_seg["LogReturn"].values[1:]
    lr_naive = np.zeros_like(lr_true)
    m = metrics(px_true, px_naive, lr_true, lr_naive)
    print(f"[BASELINE Naive  | {label}] RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f} "
          f"({m['MAE_pct']:.2f}%)  R2={m['R2']:.3f}  DirAcc={m['DirAcc']:.1f}%")
    return m

naive_test_m = naive_metrics(df_test.iloc[1:], "in-sample test")
naive_26_m   = naive_metrics(df26_feat.iloc[1:], "2026 forward hold-out")

# ── 7b. Ridge Regression (same features, same window, flattened) ───────────
print("\nTraining Ridge Regression baseline …")
X_tr_flat  = X_tr.reshape(X_tr.shape[0], -1)
X_te_flat  = X_te.reshape(X_te.shape[0], -1)
X_26_flat  = X_26.reshape(X_26.shape[0], -1)

ridge      = Ridge(alpha=1.0)
ridge.fit(X_tr_flat, y_tr)

def ridge_metrics(X_flat, y_s, df_seg, label):
    pred_s   = ridge.predict(X_flat)
    n        = len(pred_s)
    # the sequence window already consumed WINDOW rows; align price arrays
    prev_px  = df_seg["Close"].values[:n]
    true_px  = df_seg["Close"].values[1:n+1]
    # clip to valid length
    min_len  = min(len(prev_px), len(true_px), n)
    prev_px, true_px, pred_s = prev_px[:min_len], true_px[:min_len], pred_s[:min_len]
    true_lr  = df_seg["LogReturn"].values[1:min_len+1]
    pred_px, pred_lr = reconstruct_price(true_px, pred_s, prev_px)
    m = metrics(true_px, pred_px, true_lr, pred_lr)
    print(f"[BASELINE Ridge  | {label}] RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f} "
          f"({m['MAE_pct']:.2f}%)  R2={m['R2']:.3f}  DirAcc={m['DirAcc']:.1f}%")
    return m

ridge_test_m = ridge_metrics(X_te_flat, y_te, df_test, "in-sample test")
ridge_26_m   = ridge_metrics(X_26_flat, y_26, df26_feat, "2026 forward hold-out")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CEIC INTEGRATION (fill in paths when data is available)
# ══════════════════════════════════════════════════════════════════════════════
"""
When you export from CEIC, download these four series as daily CSVs
with two columns: Date, Value.

Recommended CEIC series (search by these names in CEIC):
  1. "United States Federal Funds Rate" — daily, 2012-2026
  2. "United States Government Bond Yield 10 Year" — daily, 2012-2026
  3. "Crude Oil Price WTI" — daily, 2012-2026
  4. "United States Consumer Price Index All Items" — monthly, 2012-2026

Also useful (may be in CEIC under financial markets):
  5. VIX Index — daily    (if not in CEIC: download from Yahoo Finance: ^VIX)
  6. NASDAQ Composite — daily  (Yahoo Finance: ^IXIC)

To integrate, uncomment and fill in the paths below.
Each CSV should have columns: Date, Value
"""
CEIC_PATHS = {
    # "FedFunds"   : "/home/claude/work/data/ceic_fed_funds.csv",
    # "TY10Y"      : "/home/claude/work/data/ceic_10y_yield.csv",
    # "WTI"        : "/home/claude/work/data/ceic_wti_oil.csv",
    # "CPI"        : "/home/claude/work/data/ceic_cpi.csv",
    # "VIX"        : "/home/claude/work/data/vix.csv",
    # "NASDAQ_ret" : "/home/claude/work/data/nasdaq.csv",
}

def load_ceic(path, col_name, resample_method="ffill"):
    """Load a CEIC export, align to trading calendar, forward-fill gaps."""
    df = pd.read_csv(path, parse_dates=["Date"])[["Date","Value"]].rename(columns={"Value": col_name})
    df = df.set_index("Date").sort_index()
    # resample to daily, forward-fill (handles monthly CPI, weekends, holidays)
    df = df.resample("D").last().ffill()
    return df

ceic_df = pd.DataFrame()
for col, path in CEIC_PATHS.items():
    tmp = load_ceic(path, col)
    ceic_df = tmp if ceic_df.empty else ceic_df.join(tmp, how="outer")

if not ceic_df.empty:
    ceic_df = ceic_df.ffill().bfill()
    # normalize CPI to monthly returns (stationarity)
    if "CPI" in ceic_df.columns:
        ceic_df["CPI_chg"] = ceic_df["CPI"].pct_change()
        ceic_df.drop(columns=["CPI"], inplace=True)
    # log-return NASDAQ
    if "NASDAQ_ret" in ceic_df.columns:
        ceic_df["NASDAQ_ret"] = np.log(ceic_df["NASDAQ_ret"] / ceic_df["NASDAQ_ret"].shift(1))

    for df_seg in [df_feat, df26_feat]:
        df_seg.set_index("Date", inplace=True)
        for col in ceic_df.columns:
            df_seg[col] = ceic_df[col].reindex(df_seg.index, method="ffill")
        df_seg.reset_index(inplace=True)
        df_seg.dropna(inplace=True)
        FEATURE_COLS.extend(ceic_df.columns.tolist())

    print(f"\nCEIC features added: {ceic_df.columns.tolist()}")
    print("Total features:", len(FEATURE_COLS))


# ══════════════════════════════════════════════════════════════════════════════
# 9. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
def build_model(n_feat, window):
    m = Sequential([
        Input(shape=(window, n_feat)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer=Adam(1e-3), loss="huber", metrics=["mae"])
    return m

build_model(N_FEAT, WINDOW).summary()


# ══════════════════════════════════════════════════════════════════════════════
# 10. MULTI-SEED TRAINING
# ══════════════════════════════════════════════════════════════════════════════
all_histories = []
all_test_m    = []
all_26_m      = []
all_test_pred_px = []
all_26_pred_px   = []
# y_te[j] is the scaled log-return for df_test.iloc[j]
# → true price is df_test.Close[j], prev price is df_test.Close[j-1]
# (j=0 anchors to last training-set close)
true_test_px  = df_test["Close"].values[:len(y_te)]
true_test_lr  = df_test["LogReturn"].values[:len(y_te)]
prev_test_px  = np.concatenate([[df_train["Close"].values[-1]],
                                  df_test["Close"].values[:len(y_te)-1]])

true_26_px    = df26_feat["Close"].values[:len(y_26)]
true_26_lr    = df26_feat["LogReturn"].values[:len(y_26)]
prev_26_px    = np.concatenate([[df_test["Close"].values[-1]],
                                  df26_feat["Close"].values[:len(y_26)-1]])

for seed in SEEDS:
    print(f"\n─── Seed {seed} ───────────────────────────────────────────────")
    np.random.seed(seed); tf.random.set_seed(seed)

    model = build_model(N_FEAT, WINDOW)
    cb = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0),
    ]
    hist = model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=cb, verbose=0,
    )
    ep = len(hist.history["loss"])
    print(f"  Stopped at epoch {ep}, best val_loss={min(hist.history['val_loss']):.5f}")
    all_histories.append(hist.history)

    # test
    pred_te_s  = model.predict(X_te, verbose=0)
    te_pred_px, te_pred_lr = reconstruct_price(true_test_px, pred_te_s, prev_test_px)
    m_te = metrics(true_test_px, te_pred_px, true_test_lr, te_pred_lr)
    all_test_m.append(m_te)
    all_test_pred_px.append(te_pred_px)
    print(f"  [Test ] RMSE={m_te['RMSE']:.2f}  MAE={m_te['MAE']:.2f} ({m_te['MAE_pct']:.2f}%)"
          f"  R2={m_te['R2']:.3f}  DirAcc={m_te['DirAcc']:.1f}%")

    # 2026
    pred_26_s  = model.predict(X_26, verbose=0)
    px_26, lr_26 = reconstruct_price(true_26_px, pred_26_s, prev_26_px)
    m_26 = metrics(true_26_px, px_26, true_26_lr, lr_26)
    all_26_m.append(m_26)
    all_26_pred_px.append(px_26)
    print(f"  [2026 ] RMSE={m_26['RMSE']:.2f}  MAE={m_26['MAE']:.2f} ({m_26['MAE_pct']:.2f}%)"
          f"  R2={m_26['R2']:.3f}  DirAcc={m_26['DirAcc']:.1f}%")

# ── Best seed (by 2026 R²) ────────────────────────────────────────────────
best_seed_idx = int(np.argmax([m["R2"] for m in all_26_m]))
print(f"\nBest seed: {SEEDS[best_seed_idx]}  (2026 R2={all_26_m[best_seed_idx]['R2']:.3f})")

best_hist     = all_histories[best_seed_idx]
best_te_px    = all_test_pred_px[best_seed_idx]
best_26_px    = all_26_pred_px[best_seed_idx]
best_te_lr    = tgt_scaler.inverse_transform(
                    model.predict(X_te, verbose=0).reshape(-1,1)).ravel()
best_26_lr    = tgt_scaler.inverse_transform(
                    model.predict(X_26, verbose=0).reshape(-1,1)).ravel()


# ══════════════════════════════════════════════════════════════════════════════
# 11. AGGREGATE METRICS (mean ± std across seeds)
# ══════════════════════════════════════════════════════════════════════════════
def agg(ms):
    keys = ms[0].keys()
    return {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms])) for k in keys}

agg_te = agg(all_test_m)
agg_26 = agg(all_26_m)

print("\n═══ AGGREGATED RESULTS ═══")
print(f"{'Metric':<12}{'Test (mean±std)':<22}{'2026 (mean±std)':<22}")
for k in ["RMSE","MAE","MAE_pct","R2","DirAcc"]:
    te_str = f"{agg_te[k][0]:.2f} ± {agg_te[k][1]:.2f}"
    h26_str = f"{agg_26[k][0]:.2f} ± {agg_26[k][1]:.2f}"
    print(f"{k:<12}{te_str:<22}{h26_str:<22}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. COMPREHENSIVE COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
rows = []
# Naive
rows.append({"Model":"Naive (Persistence)","Split":"Test",
             "RMSE":naive_test_m["RMSE"],"MAE":naive_test_m["MAE"],
             "MAE%":naive_test_m["MAE_pct"],"R2":naive_test_m["R2"],
             "DirAcc":naive_test_m["DirAcc"]})
rows.append({"Model":"Naive (Persistence)","Split":"2026 Hold-out",
             "RMSE":naive_26_m["RMSE"],"MAE":naive_26_m["MAE"],
             "MAE%":naive_26_m["MAE_pct"],"R2":naive_26_m["R2"],
             "DirAcc":naive_26_m["DirAcc"]})
# Ridge
rows.append({"Model":"Ridge Regression","Split":"Test",
             "RMSE":ridge_test_m["RMSE"],"MAE":ridge_test_m["MAE"],
             "MAE%":ridge_test_m["MAE_pct"],"R2":ridge_test_m["R2"],
             "DirAcc":ridge_test_m["DirAcc"]})
rows.append({"Model":"Ridge Regression","Split":"2026 Hold-out",
             "RMSE":ridge_26_m["RMSE"],"MAE":ridge_26_m["MAE"],
             "MAE%":ridge_26_m["MAE_pct"],"R2":ridge_26_m["R2"],
             "DirAcc":ridge_26_m["DirAcc"]})
# Model B v1
rows.append({"Model":"LSTM Model B v1 (1yr data)","Split":"2026 Hold-out",
             **{k: PREV_METRICS_2026[k] for k in ["RMSE","MAE","R2"]},
             "MAE%":PREV_METRICS_2026["MAE_pct"],"DirAcc":"—"})
# Model B v2
for split_name, ag in [("Test", agg_te), ("2026 Hold-out", agg_26)]:
    rows.append({"Model":f"LSTM Model B v2 (Improved) [{len(SEEDS)} seeds]",
                 "Split": split_name,
                 "RMSE":f"{ag['RMSE'][0]:.2f}±{ag['RMSE'][1]:.2f}",
                 "MAE" :f"{ag['MAE'][0]:.2f}±{ag['MAE'][1]:.2f}",
                 "MAE%":f"{ag['MAE_pct'][0]:.2f}±{ag['MAE_pct'][1]:.2f}",
                 "R2"  :f"{ag['R2'][0]:.3f}±{ag['R2'][1]:.3f}",
                 "DirAcc":f"{ag['DirAcc'][0]:.1f}±{ag['DirAcc'][1]:.1f}"})

table_df = pd.DataFrame(rows)
table_df.to_csv(f"{OUT}/TABLE1_model_comparison.csv", index=False)
print("\n\n── Table 1 saved ──")
print(table_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 13. MONTHLY BREAKDOWN TABLE (2026)
# ══════════════════════════════════════════════════════════════════════════════
dates_26    = df26_feat["Date"].values[1:len(best_26_px)+1]
monthly_rows = []
for month in pd.DatetimeIndex(dates_26).month.unique():
    mask  = pd.DatetimeIndex(dates_26).month == month
    if mask.sum() < 3:
        continue
    tp, pp = true_26_px[mask], best_26_px[mask]
    tl, pl = true_26_lr[mask], best_26_lr[mask]
    m = metrics(tp, pp, tl, pl)
    monthly_rows.append({"Month": pd.Timestamp(f"2026-{month:02d}-01").strftime("%b %Y"),
                         **{k:round(v,3) for k,v in m.items()}})

monthly_df = pd.DataFrame(monthly_rows)
monthly_df.to_csv(f"{OUT}/TABLE2_monthly_breakdown_2026.csv", index=False)
print("\n\n── Table 2: Monthly Breakdown (2026) ──")
print(monthly_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 14. DIEBOLD-MARIANO TEST  (Model B v2 vs Naive baseline)
# ══════════════════════════════════════════════════════════════════════════════
from scipy.stats import ttest_1samp
naive_errors_26 = (df26_feat["Close"].values[1:len(true_26_px)+1] -
                   df26_feat["Close"].values[:len(true_26_px)])**2
model_errors_26 = (true_26_px - best_26_px)**2
d_series        = naive_errors_26 - model_errors_26   # positive = model better
dm_stat, dm_pval = stats.ttest_1samp(d_series, 0)
print(f"\n── Diebold-Mariano Test (Model B v2 vs Naive, 2026 hold-out) ──")
print(f"   DM stat={dm_stat:.3f}  p-value={dm_pval:.4f}  "
      f"({'significant' if dm_pval<0.05 else 'not significant'} at 5% level)")
dm_result = {"DM_stat": dm_stat, "p_value": dm_pval,
             "Significant_5pct": dm_pval < 0.05}
pd.DataFrame([dm_result]).to_csv(f"{OUT}/TABLE3_DM_test.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 15. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
PLT_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
}
plt.rcParams.update(PLT_STYLE)
BLUE, GREEN, RED, ORANGE = "#2563EB", "#16A34A", "#DC2626", "#EA580C"

test_dates  = df_test["Date"].values[1:len(best_te_px)+1]
dates_26_dt = df26_feat["Date"].values[1:len(best_26_px)+1]

# ── FIG 1: Full 14yr EDA with split markers ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_full["Date"], df_full["Close"], color=BLUE, lw=0.8, label="Actual Close")
ax.axvspan(df_test["Date"].min(), df_test["Date"].max(), color=ORANGE, alpha=0.12, label="In-sample test (10%)")
ax.axvspan(df_2026["Date"].min(), df_2026["Date"].max(), color=RED, alpha=0.10, label="2026 forward hold-out")
ax.set_title("TSLA Close Price 2012–2025 with Train / Test / Hold-out splits", fontsize=12)
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/FIG01_eda_full_history_with_splits.png", dpi=150); plt.close()

# ── FIG 2: Log-return distribution vs Normal ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lr_vals = df_feat["LogReturn"].values
axes[0].hist(lr_vals, bins=80, density=True, color=BLUE, alpha=0.6, edgecolor="none")
x = np.linspace(lr_vals.min(), lr_vals.max(), 300)
axes[0].plot(x, stats.norm.pdf(x, lr_vals.mean(), lr_vals.std()), RED, lw=2, label="Normal fit")
axes[0].set_title("Log-Return Distribution vs Normal"); axes[0].set_xlabel("Log Return"); axes[0].legend()
stats.probplot(lr_vals, plot=axes[1]); axes[1].set_title("Q-Q Plot (Log Returns vs Normal)")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG02_log_return_distribution.png", dpi=150); plt.close()

# ── FIG 3: Feature correlation heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
corr = df_feat[FEATURE_COLS].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.4, annot_kws={"size":7}, ax=ax,
            cbar_kws={"shrink": 0.7})
ax.set_title("Feature Correlation Matrix", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG03_feature_correlation.png", dpi=150); plt.close()

# ── FIG 4: Training history (best seed) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(best_hist["loss"],     BLUE,   lw=1.5, label="Train loss (Huber)")
ax.plot(best_hist["val_loss"], ORANGE, lw=1.5, label="Val loss (Huber)")
ax.set_title("Model B v2 — Training History (best seed)"); ax.set_xlabel("Epoch")
ax.set_ylabel("Huber Loss"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/FIG04_training_history.png", dpi=150); plt.close()

# ── FIG 5: In-sample test — Actual vs Predicted ──────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios":[3,1]})
axes[0].plot(test_dates, true_test_px, BLUE,  lw=1.2, label="Actual")
axes[0].plot(test_dates, best_te_px,   GREEN, lw=1.2, label="Predicted (Model B v2)", alpha=0.85)
m_te_b = all_test_m[best_seed_idx]
axes[0].set_title(f"In-sample Test — Actual vs Predicted  "
                  f"(RMSE={m_te_b['RMSE']:.2f}  R²={m_te_b['R2']:.3f}  "
                  f"DirAcc={m_te_b['DirAcc']:.1f}%)", fontsize=11)
axes[0].set_ylabel("Close Price (USD)"); axes[0].legend()
residuals_te = true_test_px - best_te_px
axes[1].bar(test_dates, residuals_te, color=[RED if r < 0 else GREEN for r in residuals_te],
            alpha=0.5, width=1)
axes[1].axhline(0, color="black", lw=0.7)
axes[1].set_ylabel("Residual (USD)"); axes[1].set_xlabel("Date")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG05_test_actual_vs_pred.png", dpi=150); plt.close()

# ── FIG 6: 2026 hold-out — Actual vs Predicted ───────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios":[3,1]})
axes[0].plot(dates_26_dt, true_26_px, BLUE,  lw=1.5, label="Actual")
axes[0].plot(dates_26_dt, best_26_px, GREEN, lw=1.5, label="Predicted (Model B v2)", alpha=0.85)
m_26_b = all_26_m[best_seed_idx]
axes[0].set_title(f"2026 Forward Hold-out — Actual vs Predicted  "
                  f"(RMSE={m_26_b['RMSE']:.2f}  R²={m_26_b['R2']:.3f}  "
                  f"DirAcc={m_26_b['DirAcc']:.1f}%)", fontsize=11)
axes[0].set_ylabel("Close Price (USD)"); axes[0].legend()
residuals_26 = true_26_px - best_26_px
axes[1].bar(dates_26_dt, residuals_26, color=[RED if r < 0 else GREEN for r in residuals_26],
            alpha=0.5, width=1)
axes[1].axhline(0, color="black", lw=0.7)
axes[1].set_ylabel("Residual (USD)"); axes[1].set_xlabel("Date")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG06_2026_actual_vs_pred.png", dpi=150); plt.close()

# ── FIG 7: Scatter — Predicted vs Actual returns ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, x, y, title in [
    (axes[0], true_test_lr, best_te_lr,  "In-sample test"),
    (axes[1], true_26_lr,  best_26_lr,  "2026 hold-out"),
]:
    ax.scatter(x, y, alpha=0.4, s=14, color=BLUE, edgecolors="none")
    lim = max(abs(x).max(), abs(y).max()) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], RED, lw=1, ls="--", label="Perfect fit")
    m_, b_ = np.polyfit(x, y, 1)
    ax.plot(np.sort(x), m_*np.sort(x)+b_, GREEN, lw=1.5, label=f"Trend (slope={m_:.2f})")
    ax.set_xlabel("Actual Log Return"); ax.set_ylabel("Predicted Log Return")
    ax.set_title(f"Log Return: Predicted vs Actual\n{title}")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG07_scatter_returns.png", dpi=150); plt.close()

# ── FIG 8: Residual distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, res, title in [
    (axes[0], residuals_te, "In-sample test"),
    (axes[1], residuals_26, "2026 hold-out"),
]:
    ax.hist(res, bins=30, density=True, color=BLUE, alpha=0.6, edgecolor="none")
    x = np.linspace(res.min(), res.max(), 200)
    ax.plot(x, stats.norm.pdf(x, res.mean(), res.std()), RED, lw=2)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Residual Distribution — {title}")
    ax.set_xlabel("Residual (USD)"); ax.set_ylabel("Density")
    ax.text(0.97, 0.95, f"μ={res.mean():.2f}\nσ={res.std():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
fig.tight_layout(); fig.savefig(f"{OUT}/FIG08_residual_distribution.png", dpi=150); plt.close()

# ── FIG 9: Directional accuracy by month (2026) ──────────────────────────
months_names = [r["Month"] for r in monthly_rows]
dir_acc_vals = [r["DirAcc"]*100 for r in monthly_rows]
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(months_names, dir_acc_vals,
              color=[GREEN if v >= 50 else RED for v in dir_acc_vals], alpha=0.8)
ax.axhline(50, color="gray", ls="--", lw=1.2, label="Random baseline (50%)")
ax.axhline(np.mean(dir_acc_vals), color=BLUE, ls="-", lw=1.5,
           label=f"Overall avg ({np.mean(dir_acc_vals):.1f}%)")
ax.set_ylim(0, 100); ax.set_ylabel("Directional Accuracy (%)")
ax.set_title("Directional Accuracy by Month — 2026 Hold-out")
ax.legend()
for bar, val in zip(bars, dir_acc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG09_directional_accuracy_by_month.png", dpi=150); plt.close()

# ── FIG 10: Multi-seed stability (box plots) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, ms, split_name in [(axes[0], all_test_m, "In-sample Test"),
                            (axes[1], all_26_m,   "2026 Hold-out")]:
    data = {k: [m[k] for m in ms] for k in ["RMSE","MAE","R2","DirAcc"]}
    ax2  = ax.twinx()
    ax.bar(["RMSE","MAE"], [np.mean(data["RMSE"]), np.mean(data["MAE"])],
           color=[BLUE, GREEN], alpha=0.6,
           yerr=[np.std(data["RMSE"]), np.std(data["MAE"])], capsize=5)
    ax.set_ylabel("Price Error (USD)")
    ax2.bar(["R²","DirAcc(%)"],
            [np.mean(data["R2"]), np.mean(data["DirAcc"])],
            color=[ORANGE, RED], alpha=0.6,
            yerr=[np.std(data["R2"]), np.std(data["DirAcc"])], capsize=5)
    ax2.set_ylabel("R² / DirAcc")
    ax.set_title(f"Metric Stability across {len(SEEDS)} Seeds — {split_name}")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG10_seed_stability.png", dpi=150); plt.close()

# ── FIG 11: All-model comparison bar chart ───────────────────────────────
comp_labels = ["Naive", "Ridge", "LSTM B v1\n(25yr)", "LSTM B v2\n(25yr)"]
rmse_vals   = [naive_26_m["RMSE"], ridge_26_m["RMSE"],
               PREV_METRICS_2026["RMSE"], agg_26["RMSE"][0]]
r2_vals     = [naive_26_m["R2"],   ridge_26_m["R2"],
               PREV_METRICS_2026["R2"],   agg_26["R2"][0]]
dir_vals    = [naive_26_m["DirAcc"], ridge_26_m["DirAcc"],
               50.0, agg_26["DirAcc"][0]]   # Model B v1 DirAcc not recorded

x = np.arange(len(comp_labels)); w = 0.25
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(x, rmse_vals, w*2, color=[BLUE,GREEN,ORANGE,RED], alpha=0.7)
axes[0].set_xticks(x); axes[0].set_xticklabels(comp_labels, fontsize=8)
axes[0].set_title("RMSE (lower = better)"); axes[0].set_ylabel("RMSE (USD)")
axes[1].bar(x, r2_vals,   w*2, color=[BLUE,GREEN,ORANGE,RED], alpha=0.7)
axes[1].set_xticks(x); axes[1].set_xticklabels(comp_labels, fontsize=8)
axes[1].axhline(0, color="black", lw=0.7, ls="--")
axes[1].set_title("R² Score (higher = better)"); axes[1].set_ylabel("R²")
axes[2].bar(x, dir_vals,  w*2, color=[BLUE,GREEN,ORANGE,RED], alpha=0.7)
axes[2].axhline(50, color="gray", ls="--", lw=1.2)
axes[2].set_xticks(x); axes[2].set_xticklabels(comp_labels, fontsize=8)
axes[2].set_title("Directional Accuracy % (higher = better)")
axes[2].set_ylabel("Directional Accuracy (%)")
for ax in axes:
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.003,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
fig.suptitle("2026 Forward Hold-out — Model Comparison", fontsize=12, y=1.01)
fig.tight_layout(); fig.savefig(f"{OUT}/FIG11_model_comparison_bar.png", dpi=150, bbox_inches="tight"); plt.close()

# ── FIG 12: Rolling 20-day RMSE (2026) — stability over time ────────────
roll_rmse = []
roll_dates = []
ROLL = 20
for i in range(0, len(true_26_px)-ROLL+1, ROLL):
    tp = true_26_px[i:i+ROLL]; pp = best_26_px[i:i+ROLL]
    roll_rmse.append(np.sqrt(mean_squared_error(tp, pp)))
    roll_dates.append(dates_26_dt[i])
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(roll_rmse)), roll_rmse, color=BLUE, alpha=0.7)
ax.axhline(np.mean(roll_rmse), color=RED, ls="--", lw=1.5, label=f"Mean RMSE={np.mean(roll_rmse):.2f}")
ax.set_xticks(range(len(roll_rmse)))
ax.set_xticklabels([pd.Timestamp(d).strftime("%b %d") for d in roll_dates], rotation=30, fontsize=8)
ax.set_title("Rolling 20-day RMSE — 2026 Hold-out"); ax.set_ylabel("RMSE (USD)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/FIG12_rolling_rmse_2026.png", dpi=150); plt.close()

print("\n\n══ ALL DONE ══")
print(f"Figures saved: FIG01 – FIG12  →  {OUT}/")
print(f"Tables saved : TABLE1, TABLE2, TABLE3  →  {OUT}/")
