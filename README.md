# From noisy IoT telemetry to calibrated day-ahead dispatch

Code, inputs and results for the article *"From noisy IoT telemetry to calibrated day-ahead dispatch: an
end-to-end multi-factor forecasting pipeline for hybrid microgrids, tested on a real-NWP digital twin of three
Uzbek sites and on two operating systems"* (under review).

The pipeline turns raw microgrid telemetry into calibrated 24-h-ahead quantile forecasts of PV, wind and load,
and into a dispatch plan:

1. **PI-HAD**: physics-informed anomaly detection (physical rules, frozen-value test, cross-channel XGBoost residuals with a robust MAD threshold)
2. **XCI**: self-supervised cross-channel imputation trained on artificial single-channel and gateway-wide gaps
3. **Factor selection**: per task, by fused mutual information and grouped permutation importance
4. **MFR-MTNet**: multi-task quantile network (task-specific factor gates, shared CNN-BiLSTM, horizon-query attention, optional frozen ridge anchor chosen per task on validation data)
5. **Stacking + CQR**: per-horizon stacking with quantile XGBoost and split-conformal calibration of the 80 % interval
6. **Dispatch**: day-ahead LP (HiGHS), executed against ground truth / measured data, with an optional risk-aware battery reserve

## Repository layout

```
twin/                    digital twin of a PV-wind-BESS-diesel microgrid at Nukus, Bukhara, Navoiy (Uzbekistan)
  code/                  00_prepare ... 08_ems_sweep (one script per stage)
  data/<site>/           real inputs: NASA POWER hourly weather + Open-Meteo day-ahead NWP (2024-2026, raw downloads)
  sites/<site>/results/  all result tables (CSV/JSON) reported in the paper
  run_site.sh            reproduces one site end to end
real/                    validation on two operating systems (Section 6.8)
  kelmarsh/              Kelmarsh wind farm (UK), hourly SCADA 2024 + NWP; res/ = results
  dkasc/                 DKA Solar Centre microgrid (Alice Springs, AU), NWP only; raw data must be downloaded (see dkasc/README.md)
  realpipe.py            cleaning, all forecasting models, stacking, CQR, DM tests for a real site
  ems_real.py            dispatch replay on the real microgrid (tariff sensitivity: P_DEM)
  calib.py               interval reliability and Winkler scores
  run_real.sh            reproduces one real site end to end
figures/                 figure files used in the paper
```

## Quick start

```bash
pip install -r requirements.txt

# digital twin (lite = core baselines; full = all 15 baselines, 3 seeds, ablations - several days on 2 CPU threads)
cd twin && ./run_site.sh nukus full && ./run_site.sh bukhara lite && ./run_site.sh navoiy lite

# real data
cd ../real && ./run_real.sh kelmarsh
./run_real.sh dkasc            # after downloading the DKA files, see real/dkasc/README.md
```

For the Chronos-Bolt baseline, place `config.json` and `model.safetensors` from
<https://huggingface.co/amazon/chronos-bolt-small> in `models/chronos-bolt-small/` (or set `CHRONOS_DIR`).

All random seeds are fixed (0, 1, 2). Neural models were trained on 2 CPU threads; results on other hardware
may differ in the last digits.

## Where each result in the paper comes from

| Paper item | File |
|---|---|
| IoT cleaning (detection, imputation), twin | `twin/sites/<site>/results/detection_per_channel.csv`, `imputation_per_channel.csv`, `cleaning_summary.json` |
| Factor selection (Fig. 3) | `twin/sites/<site>/results/factor_selection.json` |
| Point and probabilistic accuracy, DM tests, twin | `twin/sites/<site>/results/accuracy_table.csv`, `dm_tests.csv`, `horizon_rmse.json` |
| Ablations, twin | `twin/sites/nukus/results/accuracy_table.csv` (rows `abl_*`) |
| EMS and reserve sweep (Fig. 6) | `twin/sites/<site>/results/ems_results.csv` (R = 6 h), `ems_sweep.csv` (Nukus, R = 2–24 h) |
| Real wind farm (Kelmarsh) | `real/kelmarsh/res/accuracy.csv`, `dm.csv`, `cleaning.json`, `eval_extra.json`; Fig. 7 in `figures/` |
| Real microgrid (DKA) | `real/dkasc/res/accuracy.csv`, `dm.csv`, `cleaning.json`, `eval_extra.json`; Fig. 8 in `figures/` |
| Real-data dispatch replay | `real/dkasc/res/ems_real.csv`, `ems_real_dem*.csv` |
| Interval reliability on real data | `real/calibration_real.csv` |

## Data sources and licences

See [`DATA_LICENSES.md`](DATA_LICENSES.md). In short: NASA POWER (public), Open-Meteo forecasts (CC BY 4.0),
Kelmarsh SCADA (CC BY 4.0, doi:10.5281/zenodo.16807551). The DKA microgrid data are available from the DKA Solar
Centre but are not redistributed here. Generation, load and faults of the digital twin are simulated by the code.

## Licence

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.

## Citation

See `CITATION.cff`. Please cite the article and the original data providers.
