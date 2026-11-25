import os
"""Day-ahead EMS replay on the real DKA microgrid (measured PV and demand, real battery size).
Plans at 00:00 for the next 24 h with each forecaster, then executes against measured values."""
import numpy as np, pandas as pd, json
from scipy.optimize import linprog
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dkasc"); H = 24
E, PB, ETA, SMIN, SMAX, S0 = 380.0, 125.0, 0.95, 0.10, 0.95, 0.50      # kWh, kW (estimated from SOC/power records)
hrs = np.arange(H)
P_IMP = np.where((hrs >= 7) & (hrs < 21), 0.32, 0.16)      # illustrative AUD/kWh, peak 07-21
P_EXP, P_DEM, P_DEG = 0.08, 0.60, 0.02                    # feed-in, daily peak-demand charge AUD/kW-day, throughput cost

d = pd.read_csv(f"{R}/clean.csv", parse_dates=["time"])
te = np.load(f"{R}/res/pred_ridge_s0.npz")["te"]; t0 = pd.DatetimeIndex(d.time.values[te + 1])
days = np.where(t0.hour == 0)[0]
act_pv = np.stack([d.p_pv.values[te[i] + 1: te[i] + 25] for i in days]).clip(0)
act_ld = np.stack([d.p_load.values[te[i] + 1: te[i] + 25] for i in days]).clip(0)

def med(m):
    import glob
    fs = sorted(glob.glob(f"{R}/res/pred_{m}_s*.npz")); p = np.mean([np.load(f)["pq"] for f in fs], 0)
    return p[..., p.shape[-1]//2] if p.ndim == 4 else p
S = np.load(f"{R}/res/framework_pq.npy")
FC = {"Persistence": med("persistence"), "Ridge": med("ridge"), "XGB-Q": med("xgbq"), "TFT": med("tft"), "Framework (P50)": S[..., 1]}

def plan(net, peak_ref=None):
    """LP: x = [c(24), dis(24), imp(24), exp(24), soc(24), peak]; net = load - pv forecast"""
    n = H; nv = 5*n + 1; c = np.zeros(nv)
    c[0:n] = P_DEG; c[n:2*n] = P_DEG; c[2*n:3*n] = P_IMP; c[3*n:4*n] = -P_EXP; c[-1] = P_DEM
    Aeq, beq = [], []
    for t in range(n):
        r = np.zeros(nv); r[t] = -1; r[n+t] = 1; r[2*n+t] = 1; r[3*n+t] = -1; Aeq.append(r); beq.append(net[t])   # dis - c + imp - exp = net
        r = np.zeros(nv); r[4*n+t] = 1; r[t] = -ETA/E; r[n+t] = 1/(ETA*E)
        if t: r[4*n+t-1] = -1
        Aeq.append(r); beq.append(S0 if t == 0 else 0)
    Aub, bub = [], []
    for t in range(n):
        r = np.zeros(nv); r[2*n+t] = 1; r[-1] = -1; Aub.append(r); bub.append(0)
    r = np.zeros(nv); r[5*n-1] = -1; Aub.append(r); bub.append(-S0)          # end SOC >= start SOC
    b = [(0, PB)]*n + [(0, PB)]*n + [(0, None)]*n + [(0, None)]*n + [(SMIN, SMAX)]*n + [(0, None)]
    res = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=b, method="highs")
    x = res.x; return x[n:2*n] - x[0:n]                                       # planned battery power (+ discharge)

def execute(bp, pv, ld):
    soc = S0; imp = np.zeros(H); exp_ = np.zeros(H); thr = 0
    for t in range(H):
        p = bp[t] if bp is not None else np.clip(ld[t] - pv[t], -PB, PB)      # rule-based: self-consumption
        p = min(p, (soc - SMIN)*E*ETA) if p > 0 else max(p, -(SMAX - soc)*E/ETA)
        p = float(np.clip(p, -PB, PB)); soc -= (p/ETA if p > 0 else p*ETA)/E; thr += abs(p)
        g = ld[t] - pv[t] - p; imp[t] = max(g, 0); exp_[t] = max(-g, 0)
    cost = (P_IMP*imp).sum() - P_EXP*exp_.sum() + P_DEM*imp.max() + P_DEG*thr
    return cost, imp.sum(), imp.max(), soc

rows = {}
def run(name, bpf):
    out = np.array([execute(bpf(j), act_pv[j], act_ld[j]) for j in range(len(days))])
    rows[name] = dict(cost=out[:, 0].sum(), import_MWh=out[:, 1].sum()/1000, mean_daily_peak_kW=out[:, 2].mean(), days=len(days))
run("No battery", lambda j: np.zeros(H))
run("Rule-based (self-consumption)", lambda j: None)
for k, f in FC.items(): run(f"LP + {k}", lambda j, f=f: plan(f[days[j], :, 1] - f[days[j], :, 0]))
run("LP + Framework, risk-aware (P90 demand, P10 PV)", lambda j: plan(S[days[j], :, 1, 2] - S[days[j], :, 0, 0]))
run("LP + perfect foresight", lambda j: plan(act_ld[j] - act_pv[j]))
T = pd.DataFrame(rows).T
base, perf = T.loc["Rule-based (self-consumption)", "cost"], T.loc["LP + perfect foresight", "cost"]
T["saving_vs_rule_%"] = 100*(1 - T.cost/base); T["share_of_perfect_saving_%"] = 100*(base - T.cost)/(base - perf)
print(T.round(2).to_string()); T.to_csv(f"{R}/res/ems_real.csv")
json.dump(dict(E=E, PB=PB, ETA=ETA, tariff="illustrative AUD; peak 07-21 0.32, off-peak 0.16, feed-in 0.08, demand 0.60/kW-day, throughput 0.02"), open(f"{R}/res/ems_real_cfg.json", "w"))
