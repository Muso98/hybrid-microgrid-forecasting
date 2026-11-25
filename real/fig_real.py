import os
import numpy as np, pandas as pd, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kelmarsh")
ex = json.load(open(f"{R}/res/eval_extra.json")); hz = ex["horizon_rmse"]
fw = np.load(f"{R}/res/pred_framework_s0.npz") if False else None
P = {m: np.load(f"{R}/res/pred_{m}_s0.npz") for m in ["ridge", "xgbq", "proposed"]}
d = pd.read_csv(f"{R}/clean.csv", parse_dates=["time"])
# rebuild framework quantiles from eval (same as realpipe eval)
import importlib.util, sys
y = P["ridge"]["y"]; te = P["ridge"]["te"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
# (a) consecutive issue times at 00 UTC over one test week
S = np.load(f"{R}/res/framework_pq.npy")
t0 = pd.DatetimeIndex(d.time.values[te + 1])
sel = np.where((t0.hour == 0))[0]
blk = sel[(t0[sel] >= t0[sel][0] + pd.Timedelta(days=21))][:7]
for j, i in enumerate(blk):
    tt = d.time.values[te[i] + 1: te[i] + 25]
    ax[0].fill_between(tt, S[i, :, 0, 0], S[i, :, 0, 2], color="#9ecae1", alpha=0.8, lw=0, label="Framework P10–P90" if j == 0 else None)
    ax[0].plot(tt, S[i, :, 0, 1], color="#08519c", lw=1.3, label="Framework P50" if j == 0 else None)
    ax[0].plot(tt, y[i, :, 0], color="k", lw=1.0, label="Measured (SCADA)" if j == 0 else None)
ax[0].set_ylabel("Farm power (MW)"); ax[0].set_ylim(0, 12.8); ax[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
ax[0].set_title("(a) Kelmarsh, seven day-ahead forecasts issued 00 UTC", fontsize=9, loc="left")
for lab, m, c, ls in [("Framework", "framework", "#08519c", "-"), ("MFR-MTNet (3-seed ens.)", "proposed_ens", "#6baed6", "--"),
                      ("Ridge", "ridge", "#e6550d", "-"), ("TFT", "tft", "#756bb1", ":"), ("Chronos-Bolt (zero-shot)", "chronos", "#bcbd22", ":"), ("XGB-Q", "xgbq", "#31a354", "-."), ("Persistence", "persistence", "grey", "-")]:
    ax[1].plot(range(1, 25), [v[0] for v in hz[m]], ls, color=c, lw=1.4, label=lab)
ax[1].set_xlabel("Lead time (h)"); ax[1].set_ylabel("RMSE (MW)"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("(b) RMSE by lead time, test set", fontsize=9, loc="left")
for a in ax: a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(f"{R}/res/fig7_kelmarsh.png", dpi=200)
print("ok", [str(t0[i]) for i in blk])
