#!/bin/bash
# Reproduce Section 6.8 (real-data validation).
# usage: ./run_real.sh kelmarsh|dkasc
# DKA raw files must be downloaded first (see dkasc/README.md); Chronos-Bolt weights go in ../models/chronos-bolt-small
set -e; DS=$1; cd "$(dirname "$0")"
python3 prep_$DS.py
python3 realpipe.py $DS clean
for m in persistence ridge xgbq lgbm chronos; do python3 realpipe.py $DS fc $m 0; done
for s in 0 1 2; do for m in proposed abl_noanchor abl_noattn abl_raw abl_nonwp nhits dlinear tft; do
  [ -f $DS/res/pred_${m}_s$s.npz ] || python3 realpipe.py $DS fc $m $s; done; done   # resumable; MFR-MTNet ~1 h/seed on 2 CPU threads
python3 realpipe.py $DS eval
if [ "$DS" = "dkasc" ]; then python3 ems_real.py; python3 fig_dkasc.py; else python3 fig_real.py; fi
python3 calib.py || true      # needs both sites
