import os
import numpy as np, pandas as pd, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dkasc")
ex = json.load(open(f"{R}/res/eval_extra.json")); hz = ex["horizon_rmse"]
p = np.load(f"{R}/res/pred_ridge_s0.npz"); y, te = p["y"], p["te"]
d = pd.read_csv(f"{R}/clean.csv", parse_dates=["time"]); S = np.load(f"{R}/res/framework_pq.npy")
t0 = pd.DatetimeIndex(d.time.values[te + 1]); sel = np.where(t0.hour == 0)[0]
start = sel[t0[sel] >= pd.Timestamp("2025-06-02")][0]; blk = [i for i in sel if 0 <= (t0[i] - t0[start]).days < 7]
fig = plt.figure(figsize=(11, 5.6)); gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 0.001])
for r, (k, lab) in enumerate([(0, "PV power (kW)"), (1, "Site demand (kW)")]):
    a = fig.add_subplot(gs[r, 0])
    for j, i in enumerate(blk):
        tt = d.time.values[te[i] + 1: te[i] + 25]
        a.fill_between(tt, S[i, :, k, 0], S[i, :, k, 2], color="#9ecae1", alpha=0.8, lw=0, label="Framework P10–P90" if j == 0 else None)
        a.plot(tt, S[i, :, k, 1], color="#08519c", lw=1.3, label="Framework P50" if j == 0 else None)
        a.plot(tt, y[i, :, k], color="k", lw=1.0, label="Measured" if j == 0 else None)
    a.set_ylabel(lab, fontsize=9); a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
    if r == 0: a.legend(loc="upper left", fontsize=8, frameon=False, ncol=3); a.set_title("(a) DKA microgrid, seven day-ahead forecasts issued 00:00 ACST", fontsize=9, loc="left")
    b = fig.add_subplot(gs[r, 1])
    for lb, m, c, ls in [("Framework", "framework", "#08519c", "-"), ("MFR-MTNet ens.", "proposed_ens", "#6baed6", "--"), ("Ridge", "ridge", "#e6550d", "-"),
                         ("XGB-Q", "xgbq", "#31a354", "-."), ("LightGBM", "lgbm", "#74c476", ":"), ("N-HiTS", "nhits", "#756bb1", ":"), ("TFT", "tft", "#d62728", "--"), ("Chronos-Bolt", "chronos", "#bcbd22", ":")]:
        if m in hz: b.plot(range(1, 25), [v[k] for v in hz[m]], ls, color=c, lw=1.3, label=lb)
    b.set_ylabel("RMSE (kW)", fontsize=9); b.tick_params(labelsize=8); b.spines[["top", "right"]].set_visible(False)
    if r == 0: b.set_title("(b) RMSE by lead time, test set", fontsize=9, loc="left"); b.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    else: b.set_xlabel("Lead time (h)", fontsize=9)
fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(f"{R}/res/fig8_dkasc.png", dpi=200); print("ok", t0[blk[0]], t0[blk[-1]])
