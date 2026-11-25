"""Publication figures (PNG, 300 dpi) for the manuscript package."""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300, "savefig.bbox": "tight"})
GREY, BLUE, ORANGE = "#8a8a8a", "#1f5fbf", "#c8641e"
truth = pd.read_csv("data/twin_truth.csv", parse_dates=["time"])
iot = pd.read_csv("data/twin_iot.csv", parse_dates=["time"])
clean = pd.read_csv("data/twin_cleaned.csv", parse_dates=["time"])

# Fig 3: imputation example during a gateway outage
masks = pd.read_csv("data/twin_masks.csv")
gw = (masks[["ghi", "p_pv", "p_load", "ws"]] == 1).all(axis=1).values
runs = np.flatnonzero(np.diff(np.r_[0, gw.astype(int), 0]))
starts, ends = runs[::2], runs[1::2]; L = ends - starts
k = int(np.argmax(np.where((L >= 18) & (L <= 40), L, 0)))
a, b = max(0, starts[k] - 36), ends[k] + 36
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
for i, (ch, lab) in enumerate([("p_pv", "PV power (kW)"), ("p_load", "Load (kW)")]):
    t = truth.time[a:b]
    ax[i].plot(t, truth[ch][a:b], color=GREY, lw=1.6, label="Ground truth")
    ax[i].plot(t, iot[ch][a:b], ".", color="k", ms=2.5, label="IoT measurement")
    gap = slice(starts[k], ends[k])
    ax[i].plot(truth.time[gap], iot[ch].interpolate(limit_direction="both")[gap], color=ORANGE, lw=1.2, ls="--", label="Linear interpolation")
    ax[i].plot(truth.time[gap], clean[ch][gap], color=BLUE, lw=1.6, label="XCI (proposed)")
    ax[i].axvspan(truth.time[starts[k]], truth.time[ends[k] - 1], color=BLUE, alpha=0.06)
    ax[i].set_ylabel(lab)
ax[0].legend(ncol=4, fontsize=7.5, loc="upper left", frameon=False, markerscale=3)
ax[0].set_title(f"Gateway outage of {L[k]} h: cross-channel imputation recovers the daily PV and load shape", loc="left", fontsize=9.5)
fig.savefig("figs/fig3_imputation_example.png"); plt.close(fig)

# Fig 4: one test week, framework quantiles
d = np.load("results/pred_framework_s0.npz"); pq, y, te = d["pq"], d["y"], d["te"]
day0 = [i for i, o in enumerate(te) if truth.time[o + 1].hour == 0]
sel = day0[60:67]
fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
for k, lab in enumerate(["PV (kW)", "Wind (kW)", "Load (kW)"]):
    tt = np.concatenate([truth.time.values[te[i] + 1:te[i] + 25] for i in sel])
    yy = np.concatenate([y[i, :, k] for i in sel]); lo = np.concatenate([pq[i, :, k, 0] for i in sel])
    md = np.concatenate([pq[i, :, k, 1] for i in sel]); hi = np.concatenate([pq[i, :, k, 2] for i in sel])
    ax[k].fill_between(tt, lo, hi, color=BLUE, alpha=0.18, lw=0, label="P10–P90 (calibrated)")
    ax[k].plot(tt, md, color=BLUE, lw=1.3, label="P50 forecast")
    ax[k].plot(tt, yy, color="k", lw=0.9, label="Ground truth"); ax[k].set_ylabel(lab)
ax[0].legend(ncol=3, fontsize=7.5, loc="upper left", frameon=False)
ax[0].set_title("Day-ahead forecasts issued at 23:00 for seven consecutive test days", loc="left", fontsize=9.5)
fig.autofmt_xdate(); fig.savefig("figs/fig4_forecast_week.png"); plt.close(fig)

# Fig 5: RMSE by horizon
hz = json.load(open("results/horizon_rmse.json"))
show = [m for m in ["persistence", "ridge", "xgb", "lstm", "transformer", "proposed", "framework"] if m in hz]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
for k, lab in enumerate(["PV", "Wind", "Load"]):
    for m in show:
        v = np.array(hz[m])[:, k]
        ax[k].plot(range(1, 25), v, color=BLUE if m == "framework" else (ORANGE if m == "proposed" else GREY),
                   lw=1.8 if m in ("framework", "proposed") else 0.9, alpha=1 if m in ("framework", "proposed") else 0.7, label=m)
    ax[k].set_title(lab, loc="left"); ax[k].set_xlabel("Lead time (h)")
ax[0].set_ylabel("RMSE (kW)"); ax[2].legend(fontsize=6.5, frameon=False)
fig.savefig("figs/fig5_rmse_by_horizon.png"); plt.close(fig)
print("figures written")
