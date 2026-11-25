"""Align NASA POWER (LST, hour-beginning means) and Open-Meteo previous-day-1 NWP (Asia/Tashkent) to
Tashkent local time, hour-beginning convention, 2024-02-01 00:00 .. 2026-01-31 23:00.
Usage: SITE=nukus python 00_prepare.py"""
import os, json, numpy as np, pandas as pd
SITE = os.environ.get("SITE", "nukus"); D = f"data/{SITE}"
n = pd.read_csv(f"{D}/nasa_{SITE}_2024_2026.csv"); n["time"] = pd.to_datetime(n.time.astype(str), format="%Y%m%d%H")
w = pd.read_csv(f"{D}/nwp_{SITE}_2024_2026.csv", parse_dates=["time"])
w.columns = [c.replace("_previous_day1", "") for c in w.columns]
# fill rare 80 m gaps from 10 m with the site's median shear ratio
ratio = (w.wind_speed_80m / w.wind_speed_10m.replace(0, np.nan)).median()
w["wind_speed_80m"] = w.wind_speed_80m.fillna(w.wind_speed_10m * ratio)
# radiation in Open-Meteo = mean of the preceding hour -> hour-beginning convention
w["ghi_hb"] = w.shortwave_radiation.shift(-1)
# NASA LST offset vs Tashkent (UTC+5): pick the shift that best aligns the two irradiance series
m = n.set_index("time").ALLSKY_SFC_SW_DWN
best = max(range(-3, 4), key=lambda s: m.corr(w.set_index("time").ghi_hb.shift(-s).reindex(m.index)))
n["time"] = n.time + pd.Timedelta(hours=best)
idx = pd.date_range("2024-02-01 00:00", "2026-01-31 23:00", freq="h")
met = n.set_index("time").reindex(idx).interpolate(limit=3).bfill().ffill()
nwp = w.set_index("time").reindex(idx).interpolate(limit=3).bfill().ffill()
met.index.name = nwp.index.name = "time"
met.to_csv(f"{D}/met.csv"); nwp[["ghi_hb", "temperature_2m", "wind_speed_80m", "wind_speed_10m", "relative_humidity_2m", "cloud_cover"]].to_csv(f"{D}/nwp.csv")
c = {"ghi": float(met.ALLSKY_SFC_SW_DWN.corr(nwp.ghi_hb)), "t": float(met.T2M.corr(nwp.temperature_2m)), "ws": float(met.WS50M.corr(nwp.wind_speed_80m))}
json.dump({"site": SITE, "nasa_shift_h": best, "shear_ratio_80_10": float(ratio), "corr_nwp_vs_nasa": c, "hours": len(idx)}, open(f"{D}/prep.json", "w"), indent=1)
print(SITE, "nasa shift", best, "corr", {k: round(v, 3) for k, v in c.items()}, len(idx))
