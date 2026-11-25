"""
Physics-based digital twin of a grid-connected PV-wind-BESS-diesel microgrid
in Nukus, Karakalpakstan (42.46N, 59.60E, 71 m a.s.l.), driven by real
NASA POWER hourly meteorology for 2023-2024. Produces:
  data/twin_truth.csv   - noise-free ground truth states
  data/twin_iot.csv     - IoT-measured channels (noise, gaps, spikes, stuck values)
  data/twin_masks.csv   - ground-truth corruption labels per channel
"""
import numpy as np, pandas as pd, pvlib, json

RNG = np.random.default_rng(2024)
import os
SITES = {"nukus": (42.46, 59.60, 71.0), "bukhara": (39.77, 64.42, 225.0), "navoiy": (40.10, 65.38, 350.0), "tashkent": (41.31, 69.28, 455.0)}
SITE = os.environ.get("SITE", "nukus"); LAT, LON, ALT = SITES[SITE]; TZ = "Asia/Tashkent"

# ----------------------------- system configuration -------------------------
CFG = dict(
    pv_kwp=150.0, pv_tilt=40.0, pv_azimuth=180.0, pv_gamma=-0.0037, pv_sys_loss=0.10, inv_eta=0.96,
    wt_rated_kw=100.0, wt_cut_in=3.0, wt_rated_ws=11.5, wt_cut_out=25.0, wt_hub_m=50.0,
    bess_kwh=300.0, bess_kw=100.0, soc_min=0.20, soc_max=0.95, eta_c=0.95, eta_d=0.95,
    dg_kw=120.0, dg_min_load=0.30, dg_fuel_a=0.246, dg_fuel_b=0.08415, tank_l=1000.0, refill_below_l=250.0,
    grid_kw=200.0,
)

met = pd.read_csv("data/met.csv", parse_dates=["time"])
met["time"] = met.time.dt.tz_localize("Etc/GMT-5")                       # Tashkent time, hour-beginning
met = met.set_index("time")
ghi, dhi = met.ALLSKY_SFC_SW_DWN.clip(lower=0), met.ALLSKY_SFC_SW_DIFF.clip(lower=0)
T, RH, WS10, WS50, WD50 = met.T2M, met.RH2M, met.WS10M, met.WS50M, met.WD50M
PS, PREC = met.PS, met.PRECTOTCORR
n = len(met)

# ----------------------------- PV subsystem ---------------------------------
mid = met.index + pd.Timedelta(minutes=30)                 # hour-average -> mid-hour sun position
sp = pvlib.solarposition.get_solarposition(mid, LAT, LON, altitude=ALT)
zen = sp.apparent_zenith.values
cosz = np.cos(np.radians(zen))
dni = np.where(cosz > 0.087, (ghi - dhi).values / np.maximum(cosz, 0.087), 0.0).clip(0, 1100)
dni_extra = pvlib.irradiance.get_extra_radiation(mid).values
poa = pvlib.irradiance.get_total_irradiance(CFG["pv_tilt"], CFG["pv_azimuth"], zen, sp.azimuth.values,
                                            dni, ghi.values, dhi.values, dni_extra=dni_extra, model="haydavies")
poa_g = np.nan_to_num(np.asarray(poa["poa_global"])).clip(0)
tp = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
t_cell = pvlib.temperature.sapm_cell(poa_g, T.values, WS10.values, **tp)
t_mod = t_cell - poa_g / 1000.0 * tp["deltaT"]

# Soiling (Aral-basin dust): daily accumulation, dust-storm jumps, rain and manual cleaning
day = met.index.normalize()
daily = pd.DataFrame({"ws": WS10, "pr": PREC, "rh": RH}).groupby(day).agg(wsmax=("ws", "max"), pr=("pr", "sum"))
soil, s, last_clean = [], 0.0, 0
for k, (d, r) in enumerate(daily.iterrows()):
    rate = 0.0025 * (1.6 if d.month in (3, 4, 5, 6) else 1.0)        # spring dust season
    s += rate * RNG.uniform(0.6, 1.4)
    if r.wsmax > 7.5:                                                  # dust event
        s += RNG.uniform(0.01, 0.03)
    if r.pr > 3.0:                                                     # rain wash
        s *= 0.25
    if k - last_clean >= 21:                                           # scheduled manual cleaning
        s, last_clean = 0.01, k
    s = min(s, 0.30)
    soil.append(s)
soil_h = pd.Series(soil, index=daily.index).reindex(day).values       # fraction of light lost
pdc = pvlib.pvsystem.pvwatts_dc(poa_g * (1 - soil_h), t_cell, CFG["pv_kwp"] * 1000, CFG["pv_gamma"])
pdc = pdc * (1 - CFG["pv_sys_loss"])
pac = pvlib.inverter.pvwatts(pdc, CFG["pv_kwp"] * 1000 / 1.1, eta_inv_nom=CFG["inv_eta"])
p_pv = np.clip(np.nan_to_num(pac) / 1000.0, 0, None)                   # kW

# ----------------------------- wind subsystem -------------------------------
rho = PS.values * 1000.0 / (287.05 * (T.values + 273.15))
v_eq = WS50.values * (rho / 1.225) ** (1 / 3)
def power_curve(v):
    c = CFG
    p = np.where(v < c["wt_cut_in"], 0.0,
        np.where(v < c["wt_rated_ws"], c["wt_rated_kw"] * (v**3 - c["wt_cut_in"]**3) / (c["wt_rated_ws"]**3 - c["wt_cut_in"]**3),
        np.where(v < c["wt_cut_out"], c["wt_rated_kw"], 0.0)))
    return p
p_wind = power_curve(v_eq) * 0.97                                       # electrical/availability losses
wt_avail = np.ones(n)                                                   # maintenance outages
for _ in range(6):
    st = RNG.integers(0, n - 72); wt_avail[st:st + RNG.integers(12, 72)] = 0
p_wind *= wt_avail

# ----------------------------- load subsystem -------------------------------
idx = met.index
hr, dow, mon = idx.hour.values, idx.dayofweek.values, idx.month.values
holidays = pd.to_datetime(["2024-01-01", "2024-03-08", "2024-03-21", "2024-03-22", "2024-04-10", "2024-04-11", "2024-05-09", "2024-06-16",
    "2024-06-17", "2024-09-01", "2024-09-02", "2024-10-01", "2024-12-08", "2024-12-31", "2025-01-01", "2025-03-08",
    "2025-03-21", "2025-03-22", "2025-03-30", "2025-03-31", "2025-05-09", "2025-06-06", "2025-06-07", "2025-09-01", "2025-10-01",
    "2025-12-08", "2025-12-31", "2026-01-01"]).date
is_hol = np.isin(idx.date, holidays)
res_wd = np.array([.55,.50,.47,.46,.48,.58,.75,.85,.78,.70,.68,.70,.74,.72,.70,.72,.80,.95,1.0,1.0,.96,.88,.76,.64])
res_we = np.array([.60,.54,.50,.48,.48,.52,.60,.72,.82,.86,.86,.86,.86,.84,.82,.82,.86,.96,1.0,1.0,.97,.90,.80,.68])
com = np.array([.15,.15,.15,.15,.15,.20,.40,.75,.95,1.0,1.0,1.0,.90,1.0,1.0,.95,.85,.60,.35,.25,.20,.18,.15,.15])
weekend = (dow == 6) | is_hol
sat = (dow == 5) & ~is_hol
residential = 70.0 * np.where(weekend, res_we[hr], res_wd[hr])
commercial = 45.0 * com[hr] * np.where(weekend, 0.25, np.where(sat, 0.6, 1.0))
t_ema = pd.Series(T.values).ewm(halflife=6).mean().values            # thermal inertia of buildings
heating = 1.8 * np.clip(8.0 - t_ema, 0, None)                         # partial electric heating
cooling = 3.2 * np.clip(t_ema - 24.0, 0, None) * np.where((hr >= 9) & (hr <= 23), 1.0, 0.55)
pump_day = pd.Series(RNG.uniform(size=len(daily)) < 0.85, index=daily.index).reindex(day).values
rain_day = pd.Series(daily.pr.values > 2.0, index=daily.index).reindex(day).values
irrig = 35.0 * ((mon >= 4) & (mon <= 9) & (hr >= 6) & (hr < 20) & pump_day & ~rain_day)
ar = np.zeros(n)
for t in range(1, n):
    ar[t] = 0.92 * ar[t - 1] + RNG.normal(0, 0.025)
p_load = (residential + commercial + heating + cooling + irrig + 12.0) * (1 + ar)
pf = np.clip(0.92 - 0.04 * (irrig > 0) - 0.02 * (cooling > 5) + RNG.normal(0, 0.005, n), 0.80, 0.98)
q_load = p_load * np.tan(np.arccos(pf))

# ----------------------------- grid availability ----------------------------
grid_ok = np.ones(n, dtype=bool)
t = 0
while t < n:
    lam = (6.0 if mon[t] in (12, 1, 2) else 2.0) / (30 * 24)        # outages per hour
    if RNG.uniform() < lam:
        dur = int(np.clip(RNG.lognormal(np.log(3), 0.7), 1, 24)); grid_ok[t:t + dur] = False; t += dur
    t += 1

# ----------------------------- rule-based EMS (actual operation) ------------
c = CFG
soc, tank = 0.6, c["tank_l"]
out = {k: np.zeros(n) for k in ["p_bess", "soc", "p_grid", "p_dg", "fuel_l", "tank_l", "p_curt", "unserved"]}
for t in range(n):
    net = p_pv[t] + p_wind[t] - p_load[t]                           # >0 surplus
    pb = pg = pd_ = curt = uns = 0.0
    if net >= 0:
        ch = min(net, c["bess_kw"], (c["soc_max"] - soc) * c["bess_kwh"] / c["eta_c"])
        pb = -ch; rest = net - ch
        if grid_ok[t]: pg = -min(rest, c["grid_kw"]); curt = rest + pg
        else: curt = rest
    else:
        need = -net
        dis = min(need, c["bess_kw"], (soc - c["soc_min"]) * c["bess_kwh"] * c["eta_d"])
        pb = dis; need -= dis
        if need > 1e-9:
            if grid_ok[t]: pg = min(need, c["grid_kw"]); need -= pg
            if need > 1e-9:
                pd_ = min(max(need, c["dg_min_load"] * c["dg_kw"]), c["dg_kw"])
                extra = pd_ - need                                    # min-load surplus -> battery
                if extra > 0:
                    chg = min(extra, c["bess_kw"] - 0, (c["soc_max"] - soc) * c["bess_kwh"] / c["eta_c"])
                    pb -= chg; curt += extra - chg
                uns = max(0.0, need - pd_)
    soc += (-min(pb, 0) * c["eta_c"] - max(pb, 0) / c["eta_d"]) / c["bess_kwh"]
    fuel = (c["dg_fuel_b"] * c["dg_kw"] + c["dg_fuel_a"] * pd_) if pd_ > 0 else 0.0
    tank -= fuel
    if tank < c["refill_below_l"]: tank = c["tank_l"]
    for k, v in zip(out, [pb, soc, pg, pd_, fuel, tank, curt, uns]): out[k][t] = v

truth = pd.DataFrame({
    "time": idx.tz_localize(None),
    "ghi": ghi.values, "dhi": dhi.values, "poa": poa_g, "t_amb": T.values, "t_mod": t_mod, "rh": RH.values,
    "ws": WS50.values, "wd": WD50.values, "ps": PS.values, "precip": PREC.values, "soiling_pct": soil_h * 100,
    "p_pv": p_pv, "p_wind": p_wind, "p_load": p_load, "q_load": q_load, "grid_ok": grid_ok.astype(int), **out})
truth.to_csv("data/twin_truth.csv", index=False)

# ----------------------------- IoT measurement layer ------------------------
chan_noise = {"ghi": ("rel", 0.02, 3.0), "t_amb": ("abs", 0.3), "t_mod": ("abs", 0.5), "rh": ("abs", 2.0),
              "ws": ("abs", 0.2), "wd": ("abs", 5.0), "ps": ("abs", 0.05), "soiling_pct": ("abs", 0.5),
              "p_pv": ("rel", 0.01, 0.2), "p_wind": ("rel", 0.01, 0.2), "p_load": ("rel", 0.01, 0.5),
              "q_load": ("rel", 0.02, 0.5), "p_bess": ("rel", 0.01, 0.2), "soc": ("abs", 0.005),
              "p_grid": ("rel", 0.01, 0.2), "p_dg": ("rel", 0.01, 0.2), "tank_l": ("abs", 3.0)}
iot = truth[["time"]].copy(); masks = truth[["time"]].copy()
gateway_down = np.zeros(n, dtype=bool)
for _ in range(25):
    st = RNG.integers(0, n - 48); gateway_down[st:st + RNG.integers(2, 49)] = True
for ch, spec in chan_noise.items():
    x = truth[ch].values.astype(float).copy()
    if spec[0] == "rel": x = x * (1 + RNG.normal(0, spec[1], n)) + RNG.normal(0, spec[2], n) * (np.abs(x) > 0)
    else: x = x + RNG.normal(0, spec[1], n)
    if ch in ("ghi", "p_pv", "p_wind", "p_load", "q_load", "rh", "soiling_pct", "tank_l", "p_dg"):
        x = np.clip(x, 0, None)
    lab = np.zeros(n, dtype=int)                                       # 0 ok,1 missing,2 spike,3 stuck
    sp_idx = RNG.choice(n, int(0.002 * n), replace=False)
    scale = np.nanstd(truth[ch].values) + 1e-6
    x[sp_idx] = x[sp_idx] + RNG.choice([-1, 1], len(sp_idx)) * RNG.uniform(4, 8, len(sp_idx)) * scale
    if ch in ("ghi", "p_pv", "p_wind", "p_load"): x[sp_idx] = np.abs(x[sp_idx])
    lab[sp_idx] = 2
    for _ in range(5):                                                  # frozen sensor episodes
        st = RNG.integers(1, n - 24); L = RNG.integers(6, 25)
        x[st:st + L] = x[st - 1]; lab[st:st + L] = 3
    miss = (RNG.uniform(size=n) < 0.005) | gateway_down
    x[miss] = np.nan; lab[miss] = 1
    iot[ch] = x; masks[ch] = lab
iot.to_csv("data/twin_iot.csv", index=False); masks.to_csv("data/twin_masks.csv", index=False)

# ----------------------------- summary --------------------------------------
E = lambda a: a.sum() / 1000 / (len(a) / 8766)                                        # MWh per year (2 yrs)
summ = dict(
    pv_MWh_yr=E(p_pv), wind_MWh_yr=E(p_wind), load_MWh_yr=E(p_load), pv_cf=p_pv.mean() / c["pv_kwp"],
    wind_cf=p_wind.mean() / c["wt_rated_kw"], load_mean_kw=p_load.mean(), load_peak_kw=p_load.max(),
    grid_import_MWh_yr=E(np.clip(out["p_grid"], 0, None)), grid_export_MWh_yr=E(-np.clip(out["p_grid"], None, 0)),
    dg_MWh_yr=E(out["p_dg"]), dg_hours=int((out["p_dg"] > 0).sum()), fuel_l_yr=out["fuel_l"].sum() / (n / 8766),
    curt_MWh_yr=E(out["p_curt"]), unserved_kWh=out["unserved"].sum(), outage_hours=int((~grid_ok).sum()),
    re_fraction=(p_pv.sum() + p_wind.sum() - out["p_curt"].sum() + 0) / p_load.sum(),
    soiling_mean_pct=soil_h.mean() * 100, soiling_max_pct=soil_h.max() * 100,
    corrupted_share={ch: float((masks[ch] > 0).mean()) for ch in chan_noise},
    cfg=CFG)
json.dump(summ, open("results/twin_summary.json", "w"), indent=1, default=float)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k not in ("cfg",)}, indent=1, default=float))
