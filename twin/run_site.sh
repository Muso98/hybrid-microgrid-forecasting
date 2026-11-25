#!/bin/bash
# Reproduce the digital-twin results for one site.
# usage: ./run_site.sh <nukus|bukhara|navoiy> [full|lite]
#   full = all 15 baselines, 3 seeds and ablations (used for Nukus in the paper)
#   lite = reduced baseline set (used for Bukhara and Navoiy)
set -e
SITE=$1; MODE=${2:-lite}
ROOT=$(cd "$(dirname "$0")" && pwd); CODE=$ROOT/code
cd "$ROOT" && SITE=$SITE python3 $CODE/00_prepare.py                      # align NASA POWER + Open-Meteo NWP
mkdir -p sites/$SITE/{data,results,logs,figs}
cp data/$SITE/met.csv data/$SITE/nwp.csv data/$SITE/prep.json sites/$SITE/data/
cd sites/$SITE
export PYTHONPATH=$CODE SITE LR=1e-3
python3 $CODE/01_digital_twin.py      > logs/twin.log 2>&1                 # simulate microgrid + IoT faults
python3 $CODE/02_cleaning.py          > logs/clean.log 2>&1                # PI-HAD + XCI and baselines
python3 $CODE/03_factor_selection.py  > logs/fs.log 2>&1                   # per-task factor selection
run(){ python3 $CODE/04_forecast.py $1 $2 >> logs/queue.log 2>&1; }
for m in persistence ridge xgb xgbq xgbq_fs lgbm; do run $m 0; done
run proposed 0; run proposed 1; run proposed 2
if [ "$MODE" = "full" ]; then
  for s in 0 1 2; do for m in dlinear nhits tide patchtst tft lstm gru cnnlstm transformer; do run $m $s; done; done
  for s in 0 1 2; do for m in abl_single abl_noattn abl_novsn abl_nofs abl_nonwp abl_raw abl_noanchor; do run $m $s; done; done
fi
python3 $CODE/06_evaluate.py > logs/eval.log 2>&1                          # stacking, CQR, metrics, DM tests
python3 $CODE/08_ems_sweep.py > logs/ems.log 2>&1                         # dispatch + reserve sweep (R = 2..24 h)
python3 $CODE/07_figures.py  > logs/fig.log 2>&1
echo "done: sites/$SITE/results"
