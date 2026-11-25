import os
import numpy as np, glob, json, pandas as pd
QS = np.array([.1, .5, .9]); out = []
for ds, tg in [("kelmarsh", ["p_farm"]), ("dkasc", ["p_pv", "p_load"])]:
    R = os.path.join(os.path.dirname(os.path.abspath(__file__)), ds); y = np.load(f"{R}/res/pred_ridge_s0.npz")["y"]
    M = {"Framework": np.load(f"{R}/res/framework_pq.npy")}
    for m, lab in [("proposed", "MFR-MTNet (anchored)"), ("tft", "TFT"), ("xgbq", "XGB-Q")]:
        M[lab] = np.mean([np.load(f)["pq"] for f in sorted(glob.glob(f"{R}/res/pred_{m}_s*.npz"))], 0)
    for k, t in enumerate(tg):
        yy = y[:, :, k]; ok = ~np.isnan(yy)
        if t == "p_pv": ok &= yy > 0.01*np.nanmax(yy)
        for lab, p in M.items():
            q = p[:, :, k]; lo, md, hi = q[..., 0][ok], q[..., 1][ok], q[..., 2][ok]; v = yy[ok]
            a = 0.2; wink = (hi - lo) + (2/a)*(lo - v)*(v < lo) + (2/a)*(v - hi)*(v > hi)
            out.append(dict(site=ds, target=t, model=lab, below_q10=100*np.mean(v < lo), below_q50=100*np.mean(v < md), below_q90=100*np.mean(v < hi),
                            PICP80=100*np.mean((v >= lo) & (v <= hi)), width80=np.mean(hi - lo), winkler80=np.mean(wink)))
T = pd.DataFrame(out); T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_real.csv"), index=False); print(T.round(1).to_string())
