"""
Stage 2: multi-factor selection. For each target, factor groups are ranked by a fused score of
(a) mutual information and (b) grouped permutation importance of a direct XGBoost forecaster
on the validation split; the number of retained groups k is chosen on validation RMSE.
"""
import numpy as np, pandas as pd, json, xgboost as xgb
from sklearn.feature_selection import mutual_info_regression
from common import *

df, truth, det, base, cs = load_frames()
tr, va, te = split_origins(df.time.values)
GROUPS_PAST = {"ghi": ["ghi"], "t_amb": ["t_amb"], "t_mod": ["t_mod"], "rh": ["rh"], "ws": ["ws"], "wd": ["wd_s", "wd_c"],
               "ps": ["ps"], "soiling": ["soiling_pct"], "p_pv": ["p_pv"], "p_wind": ["p_wind"], "p_load": ["p_load"], "q_load": ["q_load"]}
GROUPS_FUT = {"solar_geom": ["cs", "cos_zen"], "calendar": ["hs", "hc", "ds", "dc"], "day_type": ["sun", "sat", "hol"],
              "nwp_ghi": ["nwp_ghi"], "nwp_t": ["nwp_t"], "nwp_ws": ["nwp_ws"]}

def tab(origins, step=3):
    o = origins[::step]; rows = []
    for h in (1, 3, 6, 12, 18, 24):
        t = o + h; F = {"h": np.full(len(o), h)}
        for g, cols in GROUPS_PAST.items():
            for c in cols:
                F[f"{g}|{c}|l0"] = df[c].values[o]; F[f"{g}|{c}|d24"] = df[c].values[t - 24]
                F[f"{g}|{c}|m24"] = pd.Series(df[c].values).rolling(24).mean().values[o]
        for g, cols in GROUPS_FUT.items():
            for c in cols:
                if c.startswith("nwp"):
                    k = ["nwp_ghi", "nwp_t", "nwp_ws"].index(c); F[f"{g}|{c}|f"] = nwp_for(truth, cs, base, t, h)[:, k]
                else: F[f"{g}|{c}|f"] = det[c].values[t]
        Y = {y: truth[y].values[t] for y in TARGETS}
        rows.append((pd.DataFrame(F), pd.DataFrame(Y)))
    return pd.concat([r[0] for r in rows], ignore_index=True), pd.concat([r[1] for r in rows], ignore_index=True)

Xtr, Ytr = tab(tr); Xva, Yva = tab(va)
groups = list(GROUPS_PAST) + list(GROUPS_FUT)
gcols = {g: [c for c in Xtr.columns if c.startswith(g + "|")] for g in groups}
rng = np.random.default_rng(0)
res, selected = {}, {}
for y in TARGETS:
    m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, n_jobs=2, verbosity=0)
    m.fit(Xtr, Ytr[y]); base_rmse = np.sqrt(np.mean((m.predict(Xva) - Yva[y])**2))
    sub = rng.choice(len(Xtr), 6000, replace=False)
    mi_all = pd.Series(mutual_info_regression(Xtr.iloc[sub].drop(columns="h"), Ytr[y].iloc[sub], random_state=0),
                       index=Xtr.columns.drop("h"))
    pi, mi = {}, {}
    for g in groups:
        Xp = Xva.copy(); perm = rng.permutation(len(Xp)); Xp[gcols[g]] = Xva[gcols[g]].values[perm]
        pi[g] = np.sqrt(np.mean((m.predict(Xp) - Yva[y])**2)) - base_rmse
        mi[g] = mi_all[gcols[g]].max()
    pi, mi = pd.Series(pi), pd.Series(mi)
    score = 0.5*(pi.clip(lower=0)/pi.clip(lower=0).max()) + 0.5*(mi/mi.max())
    order = score.sort_values(ascending=False).index.tolist()
    curve = []
    for k in range(1, len(order) + 1):
        cols = ["h"] + sum([gcols[g] for g in order[:k]], [])
        mk = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, n_jobs=2, verbosity=0)
        mk.fit(Xtr[cols], Ytr[y]); curve.append(np.sqrt(np.mean((mk.predict(Xva[cols]) - Yva[y])**2)))
    curve = np.array(curve); best = int(np.argmin(curve))
    k_sel = int(np.argmax(curve <= curve[best]*1.01)) + 1       # smallest k within 1% of best
    selected[y] = order[:k_sel]
    res[y] = dict(mi=mi.to_dict(), perm_importance=pi.to_dict(), fused=score.to_dict(), order=order,
                  rmse_curve=curve.tolist(), k_selected=k_sel, selected=order[:k_sel], rmse_all=float(curve[-1]),
                  rmse_selected=float(curve[k_sel-1]))
    print(y, "k=", k_sel, order[:k_sel], "rmse sel/all %.3f/%.3f" % (curve[k_sel-1], curve[-1]), flush=True)
union = sorted(set(sum(selected.values(), [])))
res["union"] = union
json.dump(res, open("results/factor_selection.json", "w"), indent=1)
print("union", union)
