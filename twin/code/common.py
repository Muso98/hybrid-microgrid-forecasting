"""Shared dataset construction for forecasting experiments."""
import numpy as np, pandas as pd, pvlib

L_IN, H = 72, 24                     # 72-h look-back, 24-h ahead
TARGETS = ["p_pv", "p_wind", "p_load"]
CAP = {"p_pv": 150.0, "p_wind": 100.0, "p_load": None}
PAST_CH = ["ghi", "t_amb", "t_mod", "rh", "ws", "wd_s", "wd_c", "ps", "soiling_pct", "p_pv", "p_wind", "p_load", "q_load"]
FUT_NWP = ["nwp_ghi", "nwp_t", "nwp_ws"]
FUT_DET = ["cs", "cos_zen", "hs", "hc", "ds", "dc", "sun", "sat", "hol"]
HOL = pd.to_datetime(["2024-01-01", "2024-03-08", "2024-03-21", "2024-03-22", "2024-04-10", "2024-04-11", "2024-05-09", "2024-06-16",
    "2024-06-17", "2024-09-01", "2024-09-02", "2024-10-01", "2024-12-08", "2024-12-31", "2025-01-01", "2025-03-08",
    "2025-03-21", "2025-03-22", "2025-03-30", "2025-03-31", "2025-05-09", "2025-06-06", "2025-06-07", "2025-09-01", "2025-10-01",
    "2025-12-08", "2025-12-31", "2026-01-01"]).date


def load_frames(source="data/twin_cleaned.csv"):
    df = pd.read_csv(source, parse_dates=["time"])
    truth = pd.read_csv("data/twin_truth.csv", parse_dates=["time"])
    idx = pd.DatetimeIndex(df.time).tz_localize("Etc/GMT-5"); mid = idx + pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(42.46, 59.60, altitude=71)
    cs = loc.get_clearsky(mid, model="ineichen")["ghi"].values
    zen = loc.get_solarposition(mid).apparent_zenith.values
    df["wd_s"], df["wd_c"] = np.sin(np.radians(df.wd)), np.cos(np.radians(df.wd))
    det = pd.DataFrame({"cs": cs, "cos_zen": np.clip(np.cos(np.radians(zen)), 0, None),
        "hs": np.sin(2*np.pi*idx.hour/24), "hc": np.cos(2*np.pi*idx.hour/24),
        "ds": np.sin(2*np.pi*idx.dayofyear/365.25), "dc": np.cos(2*np.pi*idx.dayofyear/365.25),
        "sun": (idx.dayofweek == 6).astype(float), "sat": (idx.dayofweek == 5).astype(float),
        "hol": np.isin(idx.date, HOL).astype(float)})
    # real day-ahead NWP (Open-Meteo Previous Runs API, previous_day1), aligned in 00_prepare.py
    nw = pd.read_csv("data/nwp.csv", parse_dates=["time"]).set_index("time").reindex(pd.DatetimeIndex(df.time))
    base = {"ghi": nw.ghi_hb.values.clip(0), "t": nw.temperature_2m.values, "ws": nw.wind_speed_80m.values}
    return df, truth, det, base, cs


def nwp_for(truth, cs, base, tgt_idx, lead):
    """real NWP values valid at tgt_idx, issued about 24 h before the valid time (always known at the origin)"""
    return np.stack([base["ghi"][tgt_idx], base["t"][tgt_idx], base["ws"][tgt_idx]], -1)


def split_origins(time):
    """origin = last observed hour. train: targets before 2025-02-01; afterwards alternating 14-day blocks val/test."""
    n = len(time); orig = np.arange(L_IN - 1, n - H)
    tstart = pd.DatetimeIndex(time[orig + 1])
    is23 = pd.DatetimeIndex(time[orig + H]) < pd.Timestamp("2025-02-01")
    block = ((tstart - pd.Timestamp("2025-02-01")).days // 14)
    in24 = tstart >= pd.Timestamp("2025-02-01")
    tr = orig[is23]
    va = orig[in24 & (block % 2 == 0)]
    te = orig[in24 & (block % 2 == 1)]
    # drop origins whose horizon straddles a block boundary (keeps val/test targets disjoint)
    endb = ((pd.DatetimeIndex(time[orig + H]) - pd.Timestamp("2025-02-01")).days // 14)
    same = dict(zip(orig, block == endb))
    va = np.array([o for o in va if same[o]]); te = np.array([o for o in te if same[o]])
    return tr, va, te


def build_arrays(df, truth, det, base, cs, origins, past_cols=PAST_CH, use_nwp=True):
    P = df[past_cols].values.astype(np.float32); D = det[FUT_DET].values.astype(np.float32)
    off = np.arange(-L_IN + 1, 1); fut = np.arange(1, H + 1)
    Xp = P[origins[:, None] + off[None]]                                      # (N,L,Fp)
    Xp = np.concatenate([Xp, D[origins[:, None] + off[None]]], -1)
    Xf = D[origins[:, None] + fut[None]]                                      # (N,H,Fd)
    if use_nwp:
        nw = np.stack([nwp_for(truth, cs, base, origins + h, h) for h in fut], 1).astype(np.float32)
        Xf = np.concatenate([Xf, nw], -1)
    Y = truth[TARGETS].values.astype(np.float32)[origins[:, None] + fut[None]]  # evaluate against ground truth
    Yiot = df[TARGETS].values.astype(np.float32)[origins[:, None] + fut[None]] # training labels = cleaned IoT
    return Xp, Xf, Y, Yiot
