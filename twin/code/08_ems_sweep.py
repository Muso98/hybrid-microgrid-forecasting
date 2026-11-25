"""Run the day-ahead EMS (05_ems.py) for several reserve horizons R and collect results/ems_sweep.csv.
Run inside sites/<site>/ with PYTHONPATH pointing to twin/code."""
import os, subprocess, sys, pandas as pd
code = os.path.dirname(os.path.abspath(__file__)); out = []
for R in [2, 4, 6, 8, 12, 16, 24]:
    subprocess.run([sys.executable, os.path.join(code, "05_ems.py")], env={**os.environ, "RISK_H": str(R)}, check=True, stdout=subprocess.DEVNULL)
    d = pd.read_csv("results/ems_results.csv").rename(columns={"Unnamed: 0": "model"}); d["risk_h"] = R
    out.append(d)
pd.concat(out).to_csv("results/ems_sweep.csv", index=False)
subprocess.run([sys.executable, os.path.join(code, "05_ems.py")], env={**os.environ, "RISK_H": "6"}, check=True, stdout=subprocess.DEVNULL)  # main table uses R = 6 h
