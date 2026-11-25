"""
Real-data validation of the framework (cleaning + multi-task probabilistic day-ahead forecasting).
Usage:  python realpipe.py <dataset> clean
        python realpipe.py <dataset> fc <model> [seed]
        python realpipe.py <dataset> eval
Datasets are configured in CFG below. All neural models are the same architectures as in the
digital-twin study (code/04_forecast.py), generalised to K targets.
"""
import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb, pvlib
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats

DS = sys.argv[1]; MODE = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, DS); os.makedirs(f"{R}/res", exist_ok=True)
L_IN, H = 72, 24; QS = [0.1, 0.5, 0.9]

CFG = {
 "kelmarsh": dict(
    lat=52.40, lon=-0.94, alt=140, tz="UTC", start="2024-03-07", split="2024-09-01",
    targets=["p_farm"], cap={"p_farm": 12.3},
    past=["p_farm", "ws_mean", "ws_std", "wd_s", "wd_c", "tamb", "rpm"] + [f"p{k}" for k in range(1, 7)] + [f"ws{k}" for k in range(1, 7)],
    nwp=["nwp_ws100", "nwp_ws10", "nwp_t", "nwp_wd_s", "nwp_wd_c"],
    clean_ch=[f"p{k}" for k in range(1, 7)] + [f"ws{k}" for k in range(1, 7)] + ["tamb"],
    derived=lambda d: d.assign(p_farm=d[[f"p{k}" for k in range(1, 7)]].sum(axis=1, min_count=6)/1000.0,
                               ws_mean=d[[f"ws{k}" for k in range(1, 7)]].mean(axis=1),
                               ws_std=d[[f"ws{k}" for k in range(1, 7)]].std(axis=1)),
    solar=False),
 "dkasc": dict(
    lat=-23.762, lon=133.875, alt=560, tz="Australia/Darwin", start="2024-01-21", split="2025-03-01",
    targets=["p_pv", "p_load"], cap={"p_pv": None, "p_load": None},
    past=["p_pv", "p_load", "p_grid", "p_bess", "soc", "ghi", "gti", "t_amb", "rh"],
    nwp=["nwp_ghi", "nwp_t", "nwp_rh", "nwp_cc", "nwp_ws10"],
    clean_ch=["p_pv", "p_load", "p_grid", "p_bess", "ghi", "gti", "t_amb", "rh"],
    derived=lambda d: d, solar=True),
}[DS]
TG = CFG["targets"]; K = len(TG)

# ------------------------------------------------------------------ data ----
def calendar(time):
    idx = pd.DatetimeIndex(time).tz_localize(CFG["tz"]) if CFG["tz"] != "UTC" else pd.DatetimeIndex(time).tz_localize("UTC")
    mid = idx + pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(CFG["lat"], CFG["lon"], altitude=CFG["alt"])
    cs = loc.get_clearsky(mid, model="ineichen")["ghi"].values
    zen = loc.get_solarposition(mid).apparent_zenith.values
    return pd.DataFrame({"cs": cs, "cos_zen": np.clip(np.cos(np.radians(zen)), 0, None),
        "hs": np.sin(2*np.pi*idx.hour/24), "hc": np.cos(2*np.pi*idx.hour/24),
        "ds": np.sin(2*np.pi*idx.dayofyear/365.25), "dc": np.cos(2*np.pi*idx.dayofyear/365.25),
        "sun": (idx.dayofweek == 6).astype(float), "sat": (idx.dayofweek == 5).astype(float), "hol": HOLF(idx)}), zen

def HOLF(idx):
    if DS != "dkasc": return np.zeros(len(idx))
    import holidays
    hs = holidays.Australia(subdiv="NT", years=[2024, 2025, 2026])
    return np.array([d in hs for d in idx.date], float)

def load_raw():
    d = pd.read_csv(f"{R}/data.csv", parse_dates=["time"])
    d = d[d.time >= CFG["start"]].reset_index(drop=True)
    for c in CFG["nwp"]: d[c] = d[c].interpolate(limit_direction="both")
    return d

# --------------------------------------------------------------- cleaning ---
def stuck_flags(x, minrun=3):
    f = np.zeros(len(x), bool); d = np.r_[False, np.diff(x) == 0]; run = 0
    for t in range(len(x)):
        run = run + 1 if (d[t] and not np.isnan(x[t]) and abs(x[t]) > 1e-6) else 0
        if run >= minrun - 1: f[t-run:t+1] = True
    return f

def physics_flags(ch, x, d, zen):
    f = np.zeros(len(x), bool)
    if DS == "kelmarsh":
        if ch.startswith("p"):
            ws = d["ws" + ch[1:]].values
            f |= (x > 2050*1.05) | (x < -60) | ((ws < 2.5) & (x > 300))
        if ch.startswith("ws"): f |= (x < 0) | (x > 40)
        if ch == "tamb": f |= (x < -30) | (x > 45)
    else:
        if ch in ("ghi", "gti"): f |= (x < -10) | (x > 1500) | ((zen > 96) & (x > 30))
        if ch == "p_pv": f |= (x < -10) | ((zen > 96) & (x > 3)) | ((d["ghi"].values > 500) & (x < 2))
        if ch == "rh": f |= (x < 0) | (x > 104)
        if ch == "t_amb": f |= (x < -10) | (x > 50)
        if ch == "p_load": f |= (x < 0)
    return f

def residual_flags(ch, d, cal, chs, k=6.0):
    x = d[ch].values; n = len(x); s = pd.Series(x)
    feats = pd.concat([d[[c for c in chs if c != ch]].reset_index(drop=True), cal.reset_index(drop=True),
                       pd.DataFrame({"l1": s.shift(1), "f1": s.shift(-1), "l2": s.shift(2), "f2": s.shift(-2), "l24": s.shift(24)})], axis=1)
    ok = ~np.isnan(x); w = ok.copy(); flag = np.zeros(n, bool)
    for _ in range(2):
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, n_jobs=2, verbosity=0).fit(feats[w], x[w])
        r = np.full(n, np.nan); r[ok] = x[ok] - m.predict(feats[ok])
        act = ok & (np.abs(x) > 0.02*np.nanmax(np.abs(x)))
        med = np.nanmedian(r[act]); mad = 1.4826*np.nanmedian(np.abs(r[act] - med)) + 1e-9
        flag = ok & (np.abs(r - med) > k*mad); w = ok & ~flag
    return flag

def hampel(x, win=7, k=3.0):
    s = pd.Series(x); med = s.rolling(win, center=True, min_periods=1).median()
    mad = 1.4826*(s - med).abs().rolling(win, center=True, min_periods=1).median()
    return ((s - med).abs() > k*mad + 1e-9).values & ~np.isnan(x)
def sigma3(x): return (np.abs(x - np.nanmean(x)) > 3*np.nanstd(x)) & ~np.isnan(x)
def iforest(x, cont):
    s = pd.Series(x); X = pd.DataFrame({"x": s, "d1": s.diff(), "d2": s.diff(-1)}); ok = X.notna().all(axis=1).values
    f = np.zeros(len(x), bool); f[ok] = IsolationForest(n_estimators=200, contamination=cont, random_state=0).fit(X[ok]).predict(X[ok]) == -1
    return f

def pihad(d, cal, zen, chs):
    return {ch: physics_flags(ch, d[ch].values, d, zen) | stuck_flags(d[ch].values) | residual_flags(ch, d, cal, chs) for ch in chs}

def _feats(dfm, ch, cal, chs):
    s = dfm[ch]; miss = s.isna().values; n = len(s); pos = np.arange(n)
    left = pd.Series(np.where(~miss, pos, np.nan)).ffill().values; right = pd.Series(np.where(~miss, pos, np.nan)).bfill().values
    dl = np.nan_to_num(pos - left, nan=99); dr = np.nan_to_num(right - pos, nan=99)
    return pd.concat([dfm[[c for c in chs if c != ch]].reset_index(drop=True), cal.reset_index(drop=True),
        pd.DataFrame({"lin": s.interpolate(limit_direction="both").values, "dl": dl, "dr": dr, "gap": dl + dr,
                      "vl": s.ffill().values, "vr": s.bfill().values, "y24": s.shift(24).values, "t24": s.shift(-24).values})], axis=1)

def _blocks(rng, n, base, share=0.12):
    m = np.zeros(n, bool)
    while m.mean() < share:
        L = int(rng.choice([1, 2, 3, 6, 12, 24, 36])); st = rng.integers(0, n - L); m[st:st+L] = True
    return m & ~base

def xci(df, cal, chs, extra, rounds=6, seed=0):
    """self-supervised cross-channel XGBoost imputer (single-channel and gateway-wide artificial gaps)"""
    rng = np.random.default_rng(seed); out = df.copy(); hole = df[chs].isna(); n = len(df)
    for ch in chs:
        X, Y = [], []
        for r in range(rounds):
            dfm = df.copy(); m = _blocks(rng, n, hole[ch].values)
            if r % 2: dfm.loc[m, chs] = np.nan
            else: dfm.loc[m, ch] = np.nan
            X.append(_feats(dfm, ch, pd.concat([cal, extra], axis=1), chs)[m]); Y.append(df[ch].values[m])
        mdl = xgb.XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, verbosity=0)
        mdl.fit(pd.concat(X), np.concatenate(Y))
        miss = hole[ch].values
        if miss.any(): out.loc[miss, ch] = mdl.predict(_feats(df, ch, pd.concat([cal, extra], axis=1), chs)[miss])
    return out

def inject(d, chs, rng, p_spike=0.006, p_stuck=0.004):
    """semi-synthetic benchmark: known spikes and frozen runs injected into the real telemetry"""
    x = d.copy(); lab = {}
    for ch in chs:
        v = x[ch].values.copy(); n = len(v); L = np.zeros(n, bool); sd = np.nanstd(v)
        cand = np.where(~np.isnan(v) & (np.abs(v) > 0.05*np.nanmax(np.abs(v))))[0]
        sp = rng.choice(cand, int(p_spike*n), replace=False)
        v[sp] = v[sp] + rng.choice([-1, 1], len(sp))*rng.uniform(3, 6, len(sp))*sd; L[sp] = True
        nst = int(p_stuck*n/6)
        for st in rng.choice(cand[cand < n - 8], nst, replace=False):
            ln = rng.integers(4, 9); v[st:st+ln] = v[st]; L[st+1:st+ln] = True
        x[ch] = v; lab[ch] = L
    return x, lab

def run_clean():
    d = load_raw(); cal, zen = calendar(d.time); chs = CFG["clean_ch"]; rng = np.random.default_rng(0)
    extra = d[CFG["nwp"]].reset_index(drop=True)
    # (a) detection benchmark with injected faults
    xi, lab = inject(d, chs, rng)
    fl = pihad(xi, cal, zen, chs); rows = []
    for ch in chs:
        x = xi[ch].values; v = ~np.isnan(x); y = lab[ch]
        cont = float(np.clip(y.mean(), 0.001, 0.1))
        for nm, f in {"PI-HAD (proposed)": fl[ch], "Hampel": hampel(x), "3-sigma": sigma3(x), "Isolation Forest": iforest(x, cont)}.items():
            p, r, f1, _ = precision_recall_fscore_support(y[v], f[v], average="binary", zero_division=0)
            rows.append(dict(channel=ch, method=nm, precision=p, recall=r, f1=f1))
    det = pd.DataFrame(rows); det.to_csv(f"{R}/res/detection.csv", index=False)
    print(det.groupby("method")[["precision", "recall", "f1"]].mean().round(3))
    # (b) imputation benchmark: artificial gaps on observed real values
    m = {ch: _blocks(rng, len(d), d[ch].isna().values, 0.05) for ch in chs}
    dm = d.copy()
    for ch in chs: dm.loc[m[ch], ch] = np.nan
    im = {"Linear": dm[chs].interpolate(limit_direction="both"), "Forward-fill": dm[chs].ffill().bfill(),
          "XCI (proposed)": xci(dm[chs], cal, chs, extra)[chs]}
    ir = []
    for nm, o in im.items():
        for ch in chs:
            e = o[ch].values[m[ch]] - d[ch].values[m[ch]]; ir.append(dict(channel=ch, method=nm, nmae=np.mean(np.abs(e))/np.nanstd(d[ch].values)))
    imp = pd.DataFrame(ir); imp.to_csv(f"{R}/res/imputation.csv", index=False); print(imp.groupby("method").nmae.mean().round(3))
    # (c) clean the real telemetry: PI-HAD flags -> NaN -> XCI
    flr = pihad(d, cal, zen, chs); c = d.copy()
    for ch in chs: c.loc[flr[ch], ch] = np.nan
    miss0 = {ch: float(d[ch].isna().mean()) for ch in chs}
    c[chs] = xci(c[chs], cal, chs, extra)[chs]
    c = CFG["derived"](c)
    raw = d.copy(); raw[chs] = raw[chs].interpolate(limit_direction="both"); raw = CFG["derived"](raw)
    for cc in CFG["past"]:
        c[cc] = c[cc].interpolate(limit_direction="both"); raw[cc] = raw[cc].interpolate(limit_direction="both")
    c.to_csv(f"{R}/clean.csv", index=False); raw.to_csv(f"{R}/rawfilled.csv", index=False)
    json.dump({"detection_mean": det.groupby("method")[["precision", "recall", "f1"]].mean().reset_index().to_dict("records"),
               "imputation_nmae": imp.groupby("method").nmae.mean().to_dict(),
               "flag_share_real": {ch: float(flr[ch].mean()) for ch in chs}, "missing_share_real": miss0,
               "hours": len(d), "start": str(d.time.iloc[0]), "end": str(d.time.iloc[-1])}, open(f"{R}/res/cleaning.json", "w"), indent=1)

# ------------------------------------------------------------- forecasting --
def build():
    src = "rawfilled" if MODEL == "abl_raw" else "clean"
    df = pd.read_csv(f"{R}/{src}.csv", parse_dates=["time"]); meas = load_raw(); meas = CFG["derived"](meas)
    cal, zen = calendar(df.time); FD = list(cal.columns)
    use_nwp = MODEL != "abl_nonwp"
    past = CFG["past"]; P = np.concatenate([df[past].values, cal.values], 1).astype(np.float32)
    Fut = np.concatenate([cal.values] + ([df[CFG["nwp"]].values] if use_nwp else []), 1).astype(np.float32)
    n = len(df); orig = np.arange(L_IN - 1, n - H); t0 = pd.DatetimeIndex(df.time.values[orig + 1]); tend = pd.DatetimeIndex(df.time.values[orig + H])
    sp = pd.Timestamp(CFG["split"]); blk = (t0 - sp).days // 14; blke = (tend - sp).days // 14
    tr = orig[tend < sp]; post = (t0 >= sp) & (blk == blke)
    va, te = orig[post & (blk % 2 == 0)], orig[post & (blk % 2 == 1)]
    off, fut = np.arange(-L_IN + 1, 1), np.arange(1, H + 1)
    Yc = df[TG].values.astype(np.float32); Ym = meas[TG].values.astype(np.float32)   # measured (NaN where missing)
    A = lambda o: (P[o[:, None] + off], Fut[o[:, None] + fut], Yc[o[:, None] + fut], Ym[o[:, None] + fut])
    tidx = [past.index(t) for t in TG]; night = cal.cos_zen.values
    return A(tr), A(va), A(te), tr, va, te, tidx, night, ["c_"+c for c in past] + FD, FD + (CFG["nwp"] if use_nwp else [])

class Proposed(nn.Module):
    def __init__(s, Fp, Ff, d=64, attn=True):
        super().__init__(); s.attn = attn; s.mt = K > 1
        s.gp = nn.ModuleList([nn.Sequential(nn.Linear(Fp, 32), nn.ReLU(), nn.Linear(32, Fp)) for _ in range(K)])
        s.gf = nn.ModuleList([nn.Sequential(nn.Linear(Ff, 32), nn.ReLU(), nn.Linear(32, Ff)) for _ in range(K)])
        s.inp = nn.Linear(Fp, d); s.task_emb = nn.Parameter(torch.randn(K, d)*0.1)
        s.conv = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, d, 3, padding=2, dilation=2), nn.GELU())
        s.lstm = nn.LSTM(d, d//2, batch_first=True, bidirectional=True)
        s.fut = nn.Linear(Ff, d); s.hemb = nn.Parameter(torch.randn(H, d)*0.1)
        s.mha = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        s.heads = nn.ModuleList([nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, len(QS))) for _ in range(K)])
        s.log_var = nn.Parameter(torch.zeros(K)); s.lin = nn.ModuleList([nn.Linear(24*Fp + H*Ff, H) for _ in range(K)]); s.drop = nn.Dropout(0.2)
    def lin_in(s, xp, xf): return torch.cat([xp[:, -24:].flatten(1), xf.flatten(1)], 1)
    def anchor(s, xp, xf, y):
        Z = s.lin_in(xp, xf).numpy()
        for k in range(K):
            r = Ridge(alpha=10.0).fit(Z, y[:, :, k].numpy())
            s.lin[k].weight.data = torch.tensor(r.coef_, dtype=torch.float32); s.lin[k].bias.data = torch.tensor(r.intercept_, dtype=torch.float32)
            for p_ in s.lin[k].parameters(): p_.requires_grad_(False)
    def forward(s, xp, xf):
        outs = []
        for k in range(K):
            xpk = xp*torch.softmax(s.gp[k](xp), -1)*xp.shape[-1]; xfk = xf*torch.softmax(s.gf[k](xf), -1)*xf.shape[-1]
            h = s.inp(xpk) + s.task_emb[k]; h = h + s.conv(h.transpose(1, 2)).transpose(1, 2)
            enc, _ = s.lstm(h); q = s.fut(xfk) + s.hemb
            ctx = s.mha(q, enc, enc)[0] if s.attn else enc[:, -1:].expand(-1, H, -1)
            outs.append(s.lin[k](s.lin_in(xp, xf)).unsqueeze(-1) + s.heads[k](s.drop(torch.cat([q, ctx], -1))))
        return torch.stack(outs, 2)

class DLinear(nn.Module):
    def __init__(s, Fp, Ff, k=25):
        super().__init__(); s.k = k; s.lt = nn.ModuleList([nn.Linear(L_IN, H) for _ in range(K)]); s.ls = nn.ModuleList([nn.Linear(L_IN, H) for _ in range(K)]); s.fx = nn.Linear(Ff, K)
    def forward(s, xp, xf):
        y = xp[:, :, TIDX]; pad = s.k//2
        tr = nn.functional.avg_pool1d(nn.functional.pad(y.transpose(1, 2), (pad, pad), mode="replicate"), s.k, 1).transpose(1, 2)
        return (torch.stack([s.lt[c](tr[:, :, c]) + s.ls[c]((y - tr)[:, :, c]) for c in range(K)], -1) + s.fx(xf)).unsqueeze(-1)

class NHITS(nn.Module):
    def __init__(s, Fp, Ff, d=256):
        super().__init__(); s.pools, s.ncoef = [8, 4, 1], [4, 8, 24]; s.blocks = nn.ModuleList()
        for p, nc in zip(s.pools, s.ncoef):
            s.blocks.append(nn.ModuleDict({"mlp": nn.Sequential(nn.Linear(K*(L_IN//p) + H*Ff, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU()),
                                           "back": nn.Linear(d, K*L_IN), "fore": nn.Linear(d, K*nc)}))
    def forward(s, xp, xf):
        res = xp[:, :, TIDX]; fc = 0
        for b, p, nc in zip(s.blocks, s.pools, s.ncoef):
            z = nn.functional.max_pool1d(res.transpose(1, 2), p, p).flatten(1) if p > 1 else res.transpose(1, 2).flatten(1)
            h = b["mlp"](torch.cat([z, xf.flatten(1)], 1)); res = res - b["back"](h).view(-1, K, L_IN).transpose(1, 2)
            fc = fc + nn.functional.interpolate(b["fore"](h).view(-1, K, nc), size=H, mode="linear", align_corners=True).transpose(1, 2)
        return fc.unsqueeze(-1)

class GRN(nn.Module):
    def __init__(s, i, d, o=None):
        super().__init__(); o = o or d; s.l1, s.l2, s.g = nn.Linear(i, d), nn.Linear(d, 2*o), nn.GLU(); s.skip = nn.Linear(i, o) if i != o else nn.Identity(); s.n = nn.LayerNorm(o)
    def forward(s, x): return s.n(s.skip(x) + s.g(s.l2(nn.functional.elu(s.l1(x)))))
class VSN(nn.Module):
    def __init__(s, F, d):
        super().__init__(); s.W = nn.Parameter(torch.randn(F, d)*0.1); s.b = nn.Parameter(torch.zeros(F, d)); s.w = GRN(F, d, F); s.g = GRN(d, d)
    def forward(s, x): return (torch.softmax(s.w(x), -1).unsqueeze(-1)*s.g(x.unsqueeze(-1)*s.W + s.b)).sum(-2)
class TFT(nn.Module):
    def __init__(s, Fp, Ff, d=48):
        super().__init__(); s.vp, s.vf = VSN(Fp, d), VSN(Ff, d); s.enc, s.dec = nn.LSTM(d, d, batch_first=True), nn.LSTM(d, d, batch_first=True)
        s.gate1 = nn.Sequential(nn.Linear(d, 2*d), nn.GLU()); s.n1 = nn.LayerNorm(d); s.enrich = GRN(d, d)
        s.att = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1); s.gate2 = nn.Sequential(nn.Linear(d, 2*d), nn.GLU()); s.n2 = nn.LayerNorm(d); s.pw = GRN(d, d)
        s.out = nn.Linear(d, K*len(QS)); s.register_buffer("mask", torch.triu(torch.ones(L_IN + H, L_IN + H, dtype=torch.bool), 1))
    def forward(s, xp, xf):
        ep, ef = s.vp(xp), s.vf(xf); he, st = s.enc(ep); hd, _ = s.dec(ef, st)
        h = s.n1(torch.cat([ep, ef], 1) + s.gate1(torch.cat([he, hd], 1))); e = s.enrich(h)
        z = s.pw(s.n2(e + s.gate2(s.att(e, e, e, attn_mask=s.mask)[0])))[:, L_IN:]
        return s.out(z).view(-1, H, K, len(QS))

def pinball(p, y, qs):
    q = torch.tensor(qs).view(1, 1, 1, -1); e = y.unsqueeze(-1) - p; return torch.maximum(q*e, (q - 1)*e).mean((0, 1, 3))

def run_fc():
    global TIDX
    (Xp_tr, Xf_tr, Y_tr, _), (Xp_va, Xf_va, Y_va, Ym_va), (Xp_te, Xf_te, Y_te, Ym_te), tr, va, te, TIDX, night, PC, FC = build()
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)
    mu_p, sd_p = Xp_tr.reshape(-1, Xp_tr.shape[-1]).mean(0), Xp_tr.reshape(-1, Xp_tr.shape[-1]).std(0) + 1e-6
    mu_f, sd_f = Xf_tr.reshape(-1, Xf_tr.shape[-1]).mean(0), Xf_tr.reshape(-1, Xf_tr.shape[-1]).std(0) + 1e-6
    ym, ysc = Y_tr.reshape(-1, K).mean(0), Y_tr.reshape(-1, K).std(0)
    nz = lambda a, m, s: ((a - m)/s).astype(np.float32); T = torch.tensor
    t0 = time.time(); pv = None
    def tab(Xp, Xf):
        N = len(Xp); return np.concatenate([np.concatenate([np.full((N, 1), h + 1), Xp[:, -1], Xp[:, -24:].mean(1), Xp[:, -24 + h], Xf[:, h]], 1) for h in range(H)], 0)
    if MODEL == "persistence":
        Pm = pd.read_csv(f"{R}/clean.csv")[TG].values.astype(np.float32)
        f = lambda o: Pm[o[:, None] + np.arange(1, H + 1)[None] - 24][..., None]; pq, pv = f(te), f(va)
    elif MODEL == "chronos":          # zero-shot pretrained foundation model (Chronos-Bolt-small), target history only
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(os.environ.get("CHRONOS_DIR", os.path.join(HERE, "..", "models", "chronos-bolt-small")), device_map="cpu", torch_dtype=torch.float32)
        Pm = pd.read_csv(f"{R}/clean.csv")[TG].values.astype(np.float32); CTX = 512
        def f(o):
            out = np.zeros((len(o), H, K, 3), np.float32)
            for k in range(K):
                for i in range(0, len(o), 256):
                    ob = o[i:i+256]; ctx = torch.tensor(np.stack([Pm[max(0, t - CTX + 1): t + 1, k] if t >= CTX - 1 else np.r_[np.full(CTX - 1 - t, Pm[0, k]), Pm[: t + 1, k]] for t in ob]))
                    q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=QS); out[i:i+len(ob), :, k] = q.numpy()
            return out
        pq, pv = f(te), f(va)
    elif MODEL == "ridge":
        Z = lambda Xp, Xf: np.concatenate([nz(Xp, mu_p, sd_p)[:, -24:].reshape(len(Xp), -1), nz(Xf, mu_f, sd_f).reshape(len(Xp), -1)], 1)
        r = Ridge(alpha=10.0).fit(Z(Xp_tr, Xf_tr), Y_tr.reshape(len(tr), -1))
        pq, pv = [r.predict(Z(a, b)).reshape(len(a), H, K, 1) for a, b in ((Xp_te, Xf_te), (Xp_va, Xf_va))]
    elif MODEL in ("lgbm", "xgbq"):
        import lightgbm as lgb
        Xtr = tab(Xp_tr, Xf_tr); Xte, Xva = tab(Xp_te, Xf_te), tab(Xp_va, Xf_va); nq = 1 if MODEL == "lgbm" else 3
        pq = np.zeros((len(te), H, K, nq), np.float32); pv = np.zeros((len(va), H, K, nq), np.float32)
        for k in range(K):
            y = np.concatenate([Y_tr[:, h, k] for h in range(H)])
            m = (lgb.LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=SEED)
                 if MODEL == "lgbm" else xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QS), n_estimators=600, max_depth=7,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, verbosity=0, random_state=SEED)).fit(Xtr, y)
            pq[:, :, k] = m.predict(Xte).reshape(H, len(te), -1).transpose(1, 0, 2); pv[:, :, k] = m.predict(Xva).reshape(H, len(va), -1).transpose(1, 0, 2)
    else:
        Fp, Ff = Xp_tr.shape[-1], Xf_tr.shape[-1]; quant = MODEL in ("tft",) or MODEL.startswith(("proposed", "abl"))
        net = {"dlinear": DLinear, "nhits": NHITS, "tft": TFT}.get(MODEL, Proposed)(Fp, Ff) if MODEL != "abl_noattn" else Proposed(Fp, Ff, attn=False)
        xp, xf = T(nz(Xp_tr, mu_p, sd_p)), T(nz(Xf_tr, mu_f, sd_f)); yt = T((Y_tr - ym)/ysc)
        vxp, vxf = T(nz(Xp_va, mu_p, sd_p)), T(nz(Xf_va, mu_f, sd_f)); vy = T((Y_va - ym)/ysc)
        if isinstance(net, Proposed) and MODEL != "abl_noanchor": net.anchor(xp, xf, yt)
        E = 40; opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E, 2e-5); best = np.full(K, 1e9); bst = [None]*K; bad = 0
        for ep in range(E):
            net.train(); perm = torch.randperm(len(xp))
            for i in range(0, len(perm), 256):
                b = perm[i:i+256]; p = net(xp[b], xf[b])
                if quant:
                    lk = pinball(p, yt[b], QS); loss = (torch.exp(-net.log_var)*lk + net.log_var).sum() if getattr(net, "mt", False) else lk.sum()
                else: loss = ((p[..., 0] - yt[b])**2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            sch.step(); net.eval()
            with torch.no_grad():
                pvv = torch.cat([net(vxp[i:i+512], vxf[i:i+512]) for i in range(0, len(vxp), 512)]); vk = (pinball(pvv, vy, QS) if quant else ((pvv[..., 0] - vy)**2).mean((0, 1))).numpy()
            imp = False
            for k in range(K):
                if vk[k] < best[k] - 1e-6: best[k] = vk[k]; bst[k] = {a: v.clone() for a, v in net.state_dict().items()}; imp = True
            bad = 0 if imp else bad + 1
            if bad >= 10: break
        outs = []
        for XP, XF in ((Xp_te, Xf_te), (Xp_va, Xf_va)):
            o = None
            for k in range(K):
                net.load_state_dict(bst[k]); net.eval()
                with torch.no_grad():
                    ap, af = T(nz(XP, mu_p, sd_p)), T(nz(XF, mu_f, sd_f))
                    pk = np.concatenate([net(ap[i:i+512], af[i:i+512]).numpy() for i in range(0, len(ap), 512)])
                o = pk if o is None else o; o[:, :, k] = pk[:, :, k]
            outs.append(o*ysc[None, None, :, None] + ym[None, None, :, None])
        pq, pv = outs
    def fin(p, o):
        p = np.maximum(np.asarray(p, np.float32), 0)
        for k, t in enumerate(TG):
            if CFG["cap"].get(t): p[:, :, k] = np.minimum(p[:, :, k], CFG["cap"][t])
            if t == "p_pv": p[:, :, k][night[o[:, None] + np.arange(1, H + 1)[None]] <= 0] = 0
        return np.sort(p, -1)
    np.savez_compressed(f"{R}/res/pred_{MODEL}_s{SEED}.npz", pq=fin(pq, te), pv=fin(pv, va), y=Ym_te, yv=Ym_va, yvc=Y_va, te=te, va=va)
    med = fin(pq, te)[..., len(QS)//2 if np.asarray(pq).shape[-1] == 3 else 0]
    print(DS, MODEL, SEED, "RMSE", np.sqrt(np.nanmean((med - Ym_te)**2, (0, 1))).round(4), "%.0fs" % (time.time() - t0), flush=True)

# -------------------------------------------------------------- evaluation --
def dm_test(e1, e2, h=24):
    d = np.nanmean(e1**2, 1) - np.nanmean(e2**2, 1); d = d[~np.isnan(d)]; T_ = len(d); mu = d.mean()
    g = [np.mean((d[l:] - mu)*(d[:T_-l] - mu)) for l in range(h)]; v = (g[0] + 2*sum(g[1:]))/T_
    st = mu/np.sqrt(max(v, 1e-12))*np.sqrt((T_ + 1 - 2*h + h*(h - 1)/T_)/T_); return st, 2*stats.t.sf(abs(st), T_ - 1)

def run_eval():
    import glob, re
    P = {}
    for f in sorted(glob.glob(f"{R}/res/pred_*_s*.npz")):
        m, s = re.match(r".*/pred_(.+)_s(\d+)\.npz", f).groups(); P.setdefault(m, {})[int(s)] = dict(np.load(f))
    sm = lambda m, k: np.mean([P[m][s][k] for s in sorted(P[m])], 0)
    y, yv, yvc = P["ridge"][0]["y"], P["ridge"][0]["yv"], P["ridge"][0]["yvc"]
    A_te, A_va, B_te, B_va = sm("proposed", "pq"), sm("proposed", "pv"), P["xgbq"][0]["pq"], P["xgbq"][0]["pv"]
    Qa = np.array(QS); W = np.zeros((H, K)); grid = np.linspace(0, 1, 21)
    pin = lambda p, yy: np.nanmean(np.maximum(Qa*(yy[..., None] - p), (Qa - 1)*(yy[..., None] - p)))
    SEL = {}; A_te0 = A_te.copy()
    if "abl_noanchor" in P and len(P["abl_noanchor"]) == len(P["proposed"]):          # per-task anchor choice on validation
        N_te, N_va = sm("abl_noanchor", "pq"), sm("abl_noanchor", "pv")
        for k, t in enumerate(TG):
            la, ln = pin(A_va[:, :, k], yvc[:, :, k]), pin(N_va[:, :, k], yvc[:, :, k])
            SEL[t] = dict(anchored_val_pinball=float(la), unanchored_val_pinball=float(ln), choice="anchored" if la <= ln else "unanchored")
            if ln < la: A_te[:, :, k], A_va[:, :, k] = N_te[:, :, k], N_va[:, :, k]
        P["proposed_sel"] = {0: dict(pq=A_te.copy(), y=y)}
    print("anchor selection:", SEL)
    for k in range(K):
        for h in range(H):
            yy = yvc[:, h, k][:, None]
            W[h, k] = grid[int(np.argmin([np.mean(np.maximum(Qa*(yy - (w*A_va[:, h, k] + (1-w)*B_va[:, h, k])), (Qa-1)*(yy - (w*A_va[:, h, k] + (1-w)*B_va[:, h, k])))) for w in grid]))]
    S = np.sort(W[None, :, :, None]*A_te + (1 - W[None, :, :, None])*B_te, -1); Sv = np.sort(W[None, :, :, None]*A_va + (1 - W[None, :, :, None])*B_va, -1)
    cq = np.zeros((H, K))
    for k in range(K):
        for h in range(H):
            yy, lo, hi = yvc[:, h, k], Sv[:, h, k, 0], Sv[:, h, k, 2]; m = (yy > 0.01*np.nanmax(yvc[:, :, k])) if TG[k] == "p_pv" else np.ones_like(yy, bool)
            E = np.maximum(lo - yy, yy - hi)[m]; cq[h, k] = np.quantile(E, min(1.0, 0.8*(1 + 1/len(E))))
    S[..., 0] -= cq[None]; S[..., 2] += cq[None]; S = np.sort(np.maximum(S, 0), -1)
    pv_k = [k for k, t in enumerate(TG) if t == "p_pv"]
    if pv_k: S[:, :, pv_k[0]][P["proposed"][0]["pq"][:, :, pv_k[0], 1] <= 0] = 0
    np.save(f"{R}/res/framework_pq.npy", S)
    P["framework"] = {0: dict(pq=S, y=y)}; P["proposed_ens"] = {0: dict(pq=A_te0, y=y)}
    rows = []; pt = lambda p: p[..., p.shape[-1]//2]
    for m, ss in P.items():
        ms = []
        for s in sorted(ss):
            pq = ss[s]["pq"]; r = {}
            for k, t in enumerate(TG):
                e = pt(pq)[:, :, k] - y[:, :, k]; yy = y[:, :, k]; ok = ~np.isnan(yy)
                r[f"{t}_MAE"] = np.nanmean(np.abs(e)); r[f"{t}_RMSE"] = np.sqrt(np.nanmean(e**2)); r[f"{t}_nRMSE"] = 100*r[f"{t}_RMSE"]/(CFG["cap"][t] or np.nanmean(yy))
                if pq.shape[-1] == 3:
                    day = ok & ((yy > 0.01*np.nanmax(yy)) if t == "p_pv" else True)
                    r[f"{t}_PICP80"] = 100*((yy >= pq[:, :, k, 0] - 1e-6) & (yy <= pq[:, :, k, 2] + 1e-6))[day].mean()
                    pin = np.maximum(Qa*(yy[..., None] - pq[:, :, k]), (Qa - 1)*(yy[..., None] - pq[:, :, k])); r[f"{t}_Pinball"] = np.nanmean(pin)
            ms.append(r)
        dfm = pd.DataFrame(ms); rr = dfm.mean().to_dict(); rr.update({c + "_sd": v for c, v in dfm.std().to_dict().items()} if len(ms) > 1 else {})
        rr["model"] = m; rr["n_seeds"] = len(ms); rows.append(rr)
    tabl = pd.DataFrame(rows).set_index("model")
    for t in TG: tabl[f"{t}_skill"] = 100*(1 - tabl[f"{t}_RMSE"]/tabl.loc["persistence", f"{t}_RMSE"])
    tabl.to_csv(f"{R}/res/accuracy.csv"); print(tabl[[c for c in tabl.columns if c.endswith(("RMSE", "nRMSE", "PICP80", "skill")) ] + ["n_seeds"]].round(3).to_string())
    dm = []
    for b in P:
        if b == "framework": continue
        for k, t in enumerate(TG):
            st, p = dm_test(pt(P["framework"][0]["pq"])[:, :, k] - y[:, :, k], pt(P[b][0]["pq"])[:, :, k] - y[:, :, k]); dm.append(dict(model_b=b, target=t, DM=st, p=p))
    dm = pd.DataFrame(dm); dm.to_csv(f"{R}/res/dm.csv", index=False); print(dm.round(4).to_string())
    hz = {m: [np.sqrt(np.nanmean((pt(P[m][0]["pq"]) - y)[:, h, :]**2, 0)).tolist() for h in range(H)] for m in P}
    json.dump({"horizon_rmse": hz, "weights": W.tolist(), "cqr": cq.tolist(), "n_test_origins": int(len(y)), "anchor_selection": SEL}, open(f"{R}/res/eval_extra.json", "w"))

if MODE == "clean": run_clean()
elif MODE == "fc": MODEL = sys.argv[3]; SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0; run_fc()
elif MODE == "eval": run_eval()
