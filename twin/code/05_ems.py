"""
Stage 4: operational value of forecasts. Day-ahead LP scheduling of the BESS/grid/diesel using each
model's 24-h forecasts (issued 23:00 for the next day), followed by real-time execution against the
digital-twin ground truth (including unforeseen grid outages). Reports cost, fuel, unserved energy.
"""
import numpy as np, pandas as pd, json, glob, os
from scipy.optimize import linprog

C = json.load(open("results/twin_summary.json"))["cfg"]
truth = pd.read_csv("data/twin_truth.csv", parse_dates=["time"])
H = 24
hours = np.arange(H)
tariff = np.where((hours >= 17) & (hours < 22), 0.12, np.where((hours >= 7) & (hours < 17), 0.09, 0.06))  # USD/kWh (illustrative ToU)
P_EXP, FUEL_PRICE, C_SHED, C_DEG, C_CURT = 0.03, 1.0, 1.5, 0.01, 0.0
FUEL_A, FUEL_B = C["dg_fuel_a"], C["dg_fuel_b"]
SOC0 = 0.5

def plan(pv, wd, ld, reserve=None):
    """LP over 24 h. vars per hour: imp, exp, ch, dis, dg, curt, shed, soc  -> 8H"""
    nv = 8; N = nv*H; I = lambda k, t: k*H + t
    c = np.zeros(N)
    c[I(0, hours)] = tariff; c[I(1, hours)] = -P_EXP; c[I(2, hours)] = C_DEG; c[I(3, hours)] = C_DEG
    c[I(4, hours)] = FUEL_PRICE*FUEL_A; c[I(5, hours)] = C_CURT; c[I(6, hours)] = C_SHED
    Aeq, beq = [], []
    for t in hours:                                   # power balance
        r = np.zeros(N); r[[I(0, t), I(3, t), I(4, t), I(6, t)]] = 1; r[[I(1, t), I(2, t), I(5, t)]] = -1
        Aeq.append(r); beq.append(ld[t] - pv[t] - wd[t])
        r = np.zeros(N); r[I(7, t)] = 1; r[I(2, t)] = -C["eta_c"]/C["bess_kwh"]; r[I(3, t)] = 1/(C["eta_d"]*C["bess_kwh"])
        if t > 0: r[I(7, t-1)] = -1
        Aeq.append(r); beq.append(SOC0 if t == 0 else 0.0)
    lo_soc = np.full(H, C["soc_min"]) if reserve is None else np.minimum(C["soc_min"] + reserve, 0.9)
    lo_soc[-1] = max(lo_soc[-1], SOC0)
    b = [(0, C["grid_kw"])]*H + [(0, C["grid_kw"])]*H + [(0, C["bess_kw"])]*H*2 + [(0, C["dg_kw"])]*H + [(0, None)]*H*2 \
        + [(lo_soc[t], C["soc_max"]) for t in range(H)]
    res = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=b, method="highs")
    if res.status != 0: return plan(pv, wd, ld, None) if reserve is not None else np.zeros(H)
    x = res.x
    return x[I(3, hours)] - x[I(2, hours)]            # scheduled battery power (+ discharge)

def execute(pb_sched, day_idx):
    soc = SOC0; cost = fuel = uns = imp = dg_e = curt = 0.0
    for j, t in enumerate(day_idx):
        pv, wd, ld, ok = truth.p_pv[t], truth.p_wind[t], truth.p_load[t], truth.grid_ok[t] == 1
        pb = pb_sched[j]
        pb = min(pb, C["bess_kw"], (soc - C["soc_min"])*C["bess_kwh"]*C["eta_d"]) if pb > 0 else \
             -min(-pb, C["bess_kw"], (C["soc_max"] - soc)*C["bess_kwh"]/C["eta_c"])
        net = ld - pv - wd - pb                         # residual to be covered (+) or absorbed (-)
        g = d = s = cu = 0.0
        if net > 0:
            if ok: g = min(net, C["grid_kw"]); net -= g
            if net > 0:                                 # outage or grid limit: extra BESS, then diesel
                extra = min(net, C["bess_kw"] - max(pb, 0), max(0.0, (soc - C["soc_min"])*C["bess_kwh"]*C["eta_d"] - max(pb, 0)))
                pb += extra; net -= extra
                if net > 1e-9:
                    d = min(max(net, C["dg_min_load"]*C["dg_kw"]), C["dg_kw"]); s = max(0.0, net - d)
                    over = d - min(d, net)
                    if over > 0:
                        chg = min(over, max(0.0, (C["soc_max"] - soc)*C["bess_kwh"]/C["eta_c"]))
                        pb -= chg; cu += over - chg
        else:
            surplus = -net
            if ok: g = -min(surplus, C["grid_kw"]); surplus += g
            cu += surplus
        soc += (-min(pb, 0)*C["eta_c"] - max(pb, 0)/C["eta_d"])/C["bess_kwh"]
        f = (FUEL_B*C["dg_kw"] + FUEL_A*d) if d > 0 else 0.0
        h = j % 24
        cost += (tariff[h]*g if g > 0 else P_EXP*g) + FUEL_PRICE*f + C_SHED*s + C_DEG*abs(pb)
        fuel += f; uns += s; imp += max(g, 0); dg_e += d; curt += cu
    cost += max(0.0, SOC0 - soc)*C["bess_kwh"]*0.09     # value the end-of-day energy deficit
    return dict(cost=cost, fuel=fuel, unserved=uns, grid_import=imp, dg_energy=dg_e, curtail=curt)

def rule_based(day_idx):
    """no forecast: greedy self-consumption (same as the twin's EMS)"""
    pb = []
    for t in day_idx:
        net = truth.p_pv[t] + truth.p_wind[t] - truth.p_load[t]; pb.append(np.clip(-net, -C["bess_kw"], C["bess_kw"]))
    return np.array(pb)

files = {os.path.basename(f)[5:-7]: f for f in sorted(glob.glob("results/pred_*_s0.npz"))}
d0 = np.load(files["proposed"]); te = d0["te"]
day_origins = [i for i, o in enumerate(te) if truth.time[o + 1].hour == 0]
models = ["proposed"] + [m for m in files if m != "proposed" and not m.startswith("abl")]
rows = []
RISK_H = float(os.environ.get("RISK_H", 2.0))                                           # hours of outage the reserve should bridge (beyond the diesel)
variants = [(k, False) for k in ["oracle", "rule_based"] + models] + [(k, True) for k in ["proposed", "framework", "xgbq", "xgbq_fs"] if k in files]
for k, risk in variants:
    pq = None if k in ("oracle", "rule_based") else np.load(files[k])["pq"]
    tot = dict(cost=0, fuel=0, unserved=0, grid_import=0, dg_energy=0, curtail=0)
    for i in day_origins:
        o = te[i]; day = np.arange(o + 1, o + 1 + H)
        if k == "rule_based": sched = rule_based(day)
        elif k == "oracle": sched = plan(truth.p_pv.values[day], truth.p_wind.values[day], truth.p_load.values[day])
        else:
            med = pq[i, :, :, pq.shape[-1]//2]
            if risk:   # risk-aware: keep enough SOC to bridge RISK_H hours of the P90 net load not covered by the diesel
                net90 = np.clip(pq[i, :, 2, -1] - pq[i, :, 0, 0] - pq[i, :, 1, 0] - C["dg_kw"], 0, None)
                res_soc = net90*RISK_H/(C["eta_d"]*C["bess_kwh"])
                sched = plan(med[:, 0], med[:, 1], med[:, 2], reserve=res_soc)
            else: sched = plan(med[:, 0], med[:, 1], med[:, 2])
        r = execute(sched, day)
        for a in tot: tot[a] += r[a]
    tot["model"] = k + (" (risk-aware)" if risk else ""); tot["days"] = len(day_origins); rows.append(tot); print(k, {a: round(v, 1) for a, v in tot.items() if a != "model"}, flush=True)
ems = pd.DataFrame(rows).set_index("model")
ems["regret_vs_oracle_pct"] = 100*(ems.cost - ems.loc["oracle", "cost"])/ems.loc["oracle", "cost"]
ems.to_csv("results/ems_results.csv"); print(ems.round(2))
