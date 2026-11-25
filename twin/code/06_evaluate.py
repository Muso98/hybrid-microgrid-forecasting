"""Accuracy, probabilistic metrics, Diebold-Mariano tests, horizon-wise stacking."""
import numpy as np, pandas as pd, glob, json, os, re
from scipy import stats

TG = ["p_pv", "p_wind", "p_load"]; QS = np.array([0.1, 0.5, 0.9])
truth = pd.read_csv("data/twin_truth.csv")
load_mean = truth.p_load.mean()
NORM = {"p_pv": 150.0, "p_wind": 100.0, "p_load": load_mean}

def load_preds():
    P = {}
    for f in sorted(glob.glob("results/pred_*_s*.npz")):
        m, s = re.match(r"results/pred_(.+)_s(\d+)\.npz", f).groups()
        P.setdefault(m, {})[int(s)] = dict(np.load(f))
    return P

def point(pq): return pq[..., pq.shape[-1]//2]

def metrics(pq, y, day=None):
    med = point(pq); out = {}
    for k, t in enumerate(TG):
        e = med[:, :, k] - y[:, :, k]
        if t == "p_pv" and day is not None: e_day = e[day]
        out[f"{t}_MAE"] = np.abs(e).mean(); out[f"{t}_RMSE"] = np.sqrt((e**2).mean())
        out[f"{t}_nRMSE"] = 100*out[f"{t}_RMSE"]/NORM[t]
        out[f"{t}_R2"] = 1 - (e**2).sum()/((y[:, :, k] - y[:, :, k].mean())**2).sum()
        if t == "p_pv" and day is not None: out["p_pv_RMSE_day"] = np.sqrt((e_day**2).mean())
        if pq.shape[-1] == 3:
            lo, hi = pq[:, :, k, 0], pq[:, :, k, 2]; yy = y[:, :, k]
            m = day if t == "p_pv" and day is not None else np.ones_like(yy, bool)
            out[f"{t}_PICP80"] = 100*((yy >= lo - 1e-6) & (yy <= hi + 1e-6))[m].mean()
            out[f"{t}_PINAW"] = 100*(hi - lo)[m].mean()/(yy.max() - yy.min())
            pin = np.maximum(QS*(yy[..., None] - pq[:, :, k]), (QS - 1)*(yy[..., None] - pq[:, :, k]))
            out[f"{t}_Pinball"] = pin.mean()
    return out

def dm_test(e1, e2, h=24):
    """HLN-corrected Diebold-Mariano on per-origin mean squared error; HAC with h-1 lags."""
    d = (e1**2).mean(1) - (e2**2).mean(1); T = len(d); dm = d.mean()
    g = [np.mean((d[l:] - dm)*(d[:T-l] - dm)) for l in range(h)]
    v = (g[0] + 2*sum(g[1:]))/T
    stat = dm/np.sqrt(max(v, 1e-12)); k = np.sqrt((T + 1 - 2*h + h*(h - 1)/T)/T)
    stat *= k; p = 2*stats.t.sf(abs(stat), df=T - 1)
    return stat, p

P = load_preds()
import sys; sys.path.insert(0, "code"); from common import load_frames
_, _, _det, _, _ = load_frames()
_te = P["proposed"][0]["te"]; NIGHT = _det.cos_zen.values[_te[:, None] + np.arange(1, 25)[None]] <= 0
ref = P["proposed"][0]; y = ref["y"]
from_cs = None
day = y[:, :, 0] > 0.5  # daytime mask based on truth PV>0.5 kW

# ---------- horizon-adaptive stacking (weights fitted on validation only) ----------
def seed_mean(m, key="pq"): return np.mean([P[m][s][key] for s in sorted(P[m])], 0)
if "xgbq_fs" in P and all("pq_val" in P["proposed"][s] for s in P["proposed"]):
    A_te, A_va = seed_mean("proposed"), seed_mean("proposed", "pq_val")
    B_te, B_va = P["xgbq_fs"][0]["pq"], P["xgbq_fs"][0]["pq_val"]; yv = P["proposed"][0]["y_val"]
    W = np.zeros((24, 3)); grid = np.linspace(0, 1, 21)
    for k in range(3):
        for h in range(24):
            yy = yv[:, h, k][:, None]
            losses = [np.mean(np.maximum(QS*(yy - (w*A_va[:, h, k] + (1-w)*B_va[:, h, k])), (QS-1)*(yy - (w*A_va[:, h, k] + (1-w)*B_va[:, h, k])))) for w in grid]
            W[h, k] = grid[int(np.argmin(losses))]
    S = np.sort(W[None, :, :, None]*A_te + (1 - W[None, :, :, None])*B_te, -1)
    Sv = np.sort(W[None, :, :, None]*A_va + (1 - W[None, :, :, None])*B_va, -1)
    P["framework_uncal"] = {0: dict(pq=S.copy(), y=y)}
    # split-conformal calibration of the 80 % interval (CQR, Romano et al. 2019), per task and horizon, on validation
    cq = np.zeros((24, 3)); te_ = ref["te"]
    for k in range(3):
        for h in range(24):
            yy = yv[:, h, k]; lo, hi = Sv[:, h, k, 0], Sv[:, h, k, 2]
            m = (yy > 0.5) if k == 0 else np.ones_like(yy, bool)          # PV: daylight only
            if m.sum() < 30: continue
            E = np.maximum(lo - yy, yy - hi)[m]; n = len(E)
            cq[h, k] = np.quantile(E, min(1.0, 0.8*(1 + 1/n)))
    S[..., 0] = S[..., 0] - cq[None]; S[..., 2] = S[..., 2] + cq[None]
    S[..., 0] = np.maximum(S[..., 0], 0); S[:, :, 0, :][NIGHT] = 0.0   # keep night PV at zero
    S[:, :, 1] = np.minimum(S[:, :, 1], 100.0); S[:, :, 0] = np.minimum(S[:, :, 0], 150.0)
    S = np.sort(S, -1)
    json.dump({"cqr_offsets_kW": cq.tolist()}, open("results/cqr_offsets.json", "w"))
    P["framework"] = {0: dict(pq=S, y=y)}
    P["proposed_ens"] = {0: dict(pq=A_te, y=y)}
    json.dump({"weights_deep": W.tolist()}, open("results/stacking_weights.json", "w"))
    np.savez_compressed("results/pred_framework_s0.npz", pq=np.sort(S, -1), y=y, te=ref["te"])

rows = []
for m, seeds in P.items():
    ms = [metrics(seeds[s]["pq"], seeds[s]["y"], day) for s in sorted(seeds)]
    df = pd.DataFrame(ms); r = df.mean().to_dict(); r.update({k + "_sd": v for k, v in df.std().to_dict().items()} if len(ms) > 1 else {})
    r["model"] = m; r["n_seeds"] = len(ms); rows.append(r)
tab = pd.DataFrame(rows).set_index("model")
for t in TG: tab[f"{t}_skill"] = 100*(1 - tab[f"{t}_RMSE"]/tab.loc["persistence", f"{t}_RMSE"])
tab.to_csv("results/accuracy_table.csv")
cols = [f"{t}_{m}" for t in TG for m in ("MAE", "RMSE", "nRMSE", "R2")]
print(tab[cols + ["n_seeds"]].round(3).sort_values("p_load_RMSE").to_string())
pc = [c for c in tab.columns if any(x in c for x in ("PICP80", "PINAW", "Pinball")) and not c.endswith("_sd")]
print(tab[pc].dropna().round(3).to_string())

# ---------- DM tests: framework & proposed vs others ----------
dm_rows = []
for a in ("framework", "proposed"):
    if a not in P: continue
    ea = point(P[a][0]["pq"]) - y
    for b in P:
        if b == a: continue
        eb = point(P[b][0]["pq"]) - y
        for k, t in enumerate(TG):
            st, p = dm_test(ea[:, :, k], eb[:, :, k]); dm_rows.append(dict(model_a=a, model_b=b, target=t, DM=st, p=p))
dm = pd.DataFrame(dm_rows); dm.to_csv("results/dm_tests.csv", index=False)
print(dm[dm.model_a == "framework"].pivot(index="model_b", columns="target", values="DM").round(2))

# ---------- horizon-wise RMSE ----------
hz = {m: [np.sqrt(((point(P[m][0]["pq"]) - y)[:, h, :]**2).mean(0)).tolist() for h in range(24)] for m in P}
json.dump(hz, open("results/horizon_rmse.json", "w"))
meta = {}
for f in glob.glob("results/meta_*_s0.json"):
    d = json.load(open(f)); meta[d["model"]] = d
json.dump(meta, open("results/meta_all.json", "w"), indent=1)
