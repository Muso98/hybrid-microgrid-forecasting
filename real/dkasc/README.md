# DKA Solar Centre microgrid data (not redistributed)

The measured PV, site-demand, grid and battery data of the DKP Microgrid / BESS (Desert Knowledge Precinct,
Alice Springs, Australia) are provided free of charge by the DKA Solar Centre under its terms of use and are
**not included** in this repository.

1. Open <https://dkasolarcentre.com.au/download?location=dkp-microgrid>.
2. Under *Specific Data Download* choose each of **BESS**, **Site Demand**, **Solar** and **Grid**,
   resolution **5 Minutes**, date range **2023-12-07 – 2025-09-04**, and accept the terms.
3. Save the four CSV files (`239-Site_DKA_Totals-BESS_1.csv`, `240-Site_DKA_Totals-Site_Demand.csv`,
   `241-Site_DKA_Totals-PV.csv`, `242-Site_DKA_Totals-Grid.csv`) in `real/dkasc/raw/`
   (or set `DKASC_RAW=/path/to/folder`).
4. Run `./run_real.sh dkasc`.

Please cite: Desert Knowledge Australia Centre. Download data: DKP Microgrid / BESS. Alice Springs.
http://dkasolarcentre.com.au/download (accessed 24 November 2026).

The Open-Meteo day-ahead forecasts for the site (`nwp_alice_*.csv`, CC BY 4.0) are included.
