"""
Stage 1 of the framework: physics-informed hybrid anomaly detection (PI-HAD) and
gradient-boosted cross-channel imputation (XCI) for IoT telemetry.
Evaluated against the digital-twin ground truth; compared with Hampel, Isolation Forest,
3-sigma (detection) and forward-fill, linear, KNN (imputation).
"""
import numpy as np, pandas as pd, json, pvlib, xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_recall_fscore_support

truth = pd.read_csv("data/twin_truth.csv", parse_dates=["time"])
iot = pd.read_csv("data/twin_iot.csv", parse_dates=["time"])
masks = pd.read_csv("data/twin_masks.csv", parse_dates=["time"])
CH = ["ghi", "t_amb", "t_mod", "rh", "ws", "wd", "ps", "soiling_pct", "p_pv", "p_wind", "p_load", "q_load"]  # forecasting-relevant channels
n = len(iot)
idx = pd.DatetimeIndex(iot.time).tz_localize("Etc/GMT-5")
mid = idx + pd.Timedelta(minutes=30)
loc = pvlib.location.Location(42.46, 59.60, altitude=71)
cs = loc.get_clearsky(mid, model="ineichen")["ghi"].values
zen = loc.get_solarposition(mid).apparent_zenith.values
cal = pd.DataFrame({"hs": np.sin(2*np.pi*idx.hour/24), "hc": np.cos(2*np.pi*idx.hour/24),
                    "ds": np.sin(2*np.pi*idx.dayofyear/365.25), "dc": np.cos(2*np.pi*idx.dayofyear/365.25),
                    "cs": cs, "zen": zen})
RATED = {"p_pv": 150*1.0, "p_wind": 100.0, "ghi": 1.15}  # bounds used by rules

# ======================= detection =========================================
def stuck_flags(x):
    f = np.zeros(len(x), bool); d = np.diff(x, prepend=np.nan) == 0
    run = 0
    for t in range(len(x)):
        run = run + 1 if (d[t] and not np.isnan(x[t]) and abs(x[t]) > 1e-9) else 0
        if run >= 2: f[t-run:t+1] = True
    return f

def physics_flags(ch, x, df):
    f = np.zeros(len(x), bool)
    if ch == "ghi": f |= ((zen > 96) & (x > 30)) | (x > np.maximum(1.5*cs, cs + 150))
    if ch == "p_pv":
        f |= (x > 150*1.02) | ((zen > 96) & (x > 3.0))
        g = df["ghi"].values; ok = ~np.isnan(g)
        f |= ok & (g > 300) & (x < 1.0)           # PV dead while irradiance high
    if ch == "p_wind": f |= (x > 100*1.02) | ((df["ws"].values < 2.0) & (x > 15))
    if ch == "rh": f |= (x > 104) | (x < 0)
    if ch == "soc": f |= (x > 1.0) | (x < 0.0)
    if ch == "soiling_pct": f |= (x > 35) | (x < -1)
    if ch in ("t_amb",): f |= (x < -35) | (x > 50)
    return f

def xgb_residual_flags(ch, df, k=6.0):
    """cross-channel + temporal-context model; robust MAD threshold on residuals (2 passes)"""
    x = df[ch].values
    others = df.drop(columns=[ch])
    s = pd.Series(x)
    feats = pd.concat([others.reset_index(drop=True), cal,
                       pd.DataFrame({"lag1": s.shift(1), "lead1": s.shift(-1), "lag2": s.shift(2), "lead2": s.shift(-2),
                                     "lag24": s.shift(24)})], axis=1)
    ok = ~np.isnan(x); w = ok.copy(); flag = np.zeros(n, bool)
    for _ in range(2):
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, n_jobs=2, verbosity=0)
        m.fit(feats[w], x[w])
        r = np.full(n, np.nan); r[ok] = x[ok] - m.predict(feats[ok])
        if ch == "wd": r = (r + 180) % 360 - 180                     # circular residual
        act = ok & (np.abs(x) > 0.02*np.nanmax(np.abs(x)))            # exclude structural zeros (night, calm)
        med = np.nanmedian(r[act]); mad = 1.4826*np.nanmedian(np.abs(r[act] - med)) + 1e-9
        flag = ok & (np.abs(r - med) > k*mad)
        w = ok & ~flag
    return flag

def hampel(x, win=7, k=3.0):
    s = pd.Series(x); med = s.rolling(win, center=True, min_periods=1).median()
    mad = 1.4826*(s - med).abs().rolling(win, center=True, min_periods=1).median()
    return ((s - med).abs() > k*mad + 1e-9).values & ~np.isnan(x)

def sigma3(x):
    m, sd = np.nanmean(x), np.nanstd(x); return (np.abs(x - m) > 3*sd) & ~np.isnan(x)

def iforest(x):
    s = pd.Series(x); X = pd.DataFrame({"x": s, "d1": s.diff(), "d2": s.diff(-1)})
    ok = X.notna().all(axis=1).values; f = np.zeros(n, bool)
    clf = IsolationForest(n_estimators=200, contamination=0.012, random_state=0).fit(X[ok])
    f[ok] = clf.predict(X[ok]) == -1; return f

det_res, flags_all = [], {}
for ch in CH:
    x = iot[ch].values; lab = masks[ch].values
    y = (lab == 2) | (lab == 3); valid = lab != 1           # evaluate on non-missing points
    f_phys = physics_flags(ch, x, iot); f_stuck = stuck_flags(x); f_xgb = xgb_residual_flags(ch, iot[CH])
    methods = {"PI-HAD (proposed)": f_phys | f_stuck | f_xgb, "Hampel": hampel(x), "3-sigma": sigma3(x),
               "Isolation Forest": iforest(x)}
    flags_all[ch] = methods["PI-HAD (proposed)"]
    for mname, f in methods.items():
        p, r, f1, _ = precision_recall_fscore_support(y[valid], f[valid], average="binary", zero_division=0)
        det_res.append(dict(channel=ch, method=mname, precision=p, recall=r, f1=f1))
    print(ch, "done", flush=True)
det = pd.DataFrame(det_res); det.to_csv("results/detection_per_channel.csv", index=False)
det_sum = det.groupby("method")[["precision", "recall", "f1"]].mean().sort_values("f1", ascending=False)
print(det_sum.round(3))

# ======================= imputation ========================================
cleaned_in = iot[CH].copy()
for ch in CH: cleaned_in.loc[flags_all[ch], ch] = np.nan
hole = cleaned_in.isna()

def _feats(dfm, ch):
    s = dfm[ch]; miss = s.isna().values
    lin = s.interpolate(limit_direction="both").values
    pos = np.arange(n)
    left = pd.Series(np.where(~miss, pos, np.nan)).ffill().values
    right = pd.Series(np.where(~miss, pos, np.nan)).bfill().values
    dl = np.nan_to_num(pos - left, nan=99); dr = np.nan_to_num(right - pos, nan=99)
    vl = s.ffill().values; vr = s.bfill().values
    return pd.concat([dfm.drop(columns=[ch]).reset_index(drop=True), cal,
        pd.DataFrame({"lin": lin, "dl": dl, "dr": dr, "gap": dl + dr, "vl": vl, "vr": vr,
                      "y24": s.shift(24).values, "t24": s.shift(-24).values, "y48": s.shift(48).values})], axis=1)

def _random_blocks(rng, base_mask, share=0.12):
    m = np.zeros(n, bool)
    while m.mean() < share:
        L = int(rng.choice([1, 2, 3, 6, 12, 24, 36, 48])); st = rng.integers(0, n - L); m[st:st + L] = True
    return m & ~base_mask

def xci_impute(df, rounds=6, seed=0):
    """Self-supervised cross-channel imputation: learn to fill artificial gaps (single-channel and
    gateway-wide) on observed data, then fill the real gaps."""
    rng = np.random.default_rng(seed); out = df.copy(); hole = df.isna()
    for ch in CH:
        X, Y = [], []
        for r in range(rounds):
            dfm = df.copy(); m = _random_blocks(rng, hole[ch].values)
            if r % 2 == 1:                                   # gateway-wide outage: all channels masked
                dfm.loc[m, CH] = np.nan
            else:
                dfm.loc[m, ch] = np.nan
            F = _feats(dfm, ch); X.append(F[m]); Y.append(df[ch].values[m])
        mdl = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, n_jobs=2, verbosity=0)
        mdl.fit(pd.concat(X), np.concatenate(Y))
        miss = hole[ch].values
        if miss.any(): out.loc[miss, ch] = mdl.predict(_feats(df, ch)[miss])
        print("imputed", ch, flush=True)
    return out

imps = {"Forward-fill": cleaned_in.ffill().bfill(), "Linear": cleaned_in.interpolate(limit_direction="both"),
        "KNN (k=5)": pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(pd.concat([cleaned_in, cal], axis=1))[:, :len(CH)], columns=CH),
        "XCI (proposed)": xci_impute(cleaned_in)}
imp_res = []
for mname, df in imps.items():
    for ch in CH:
        h = hole[ch].values; sc = truth[ch].std() if ch in truth else 1
        e = df[ch].values[h] - truth[ch].values[h]
        imp_res.append(dict(channel=ch, method=mname, mae=np.mean(np.abs(e)), nmae=np.mean(np.abs(e))/sc, n=int(h.sum())))
imp = pd.DataFrame(imp_res); imp.to_csv("results/imputation_per_channel.csv", index=False)
print(imp.groupby("method").nmae.mean().sort_values().round(4))
print(imp[imp.channel.isin(["p_pv", "p_wind", "p_load", "ghi", "ws", "t_amb"])].pivot(index="channel", columns="method", values="mae").round(3))

clean = imps["XCI (proposed)"]; clean.insert(0, "time", iot.time)
for ch in ("ghi", "p_pv", "p_wind", "p_load", "rh", "soiling_pct"): clean[ch] = clean[ch].clip(lower=0)
clean.to_csv("data/twin_cleaned.csv", index=False)
# raw-ish version for ablation: only linear interpolation of missing, no anomaly removal
raw = iot[CH].interpolate(limit_direction="both"); raw.insert(0, "time", iot.time); raw.to_csv("data/twin_rawfilled.csv", index=False)

json.dump({"detection_mean": det_sum.reset_index().to_dict("records"),
           "imputation_mean_nmae": imp.groupby("method").nmae.mean().to_dict(),
           "flagged_share": {ch: float(flags_all[ch].mean()) for ch in CH},
           "holes_share": float(hole.values.mean())}, open("results/cleaning_summary.json", "w"), indent=1)
