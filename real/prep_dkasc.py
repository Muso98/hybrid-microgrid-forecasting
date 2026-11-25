import os
import pandas as pd, numpy as np, glob
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dkasc')
U = os.environ.get('DKASC_RAW', os.path.join(D, 'raw')) + '/'
def rd(pat):
    f=glob.glob(U+pat)[0]; d=pd.read_csv(f, parse_dates=['timestamp'], on_bad_lines='skip', engine='python').set_index('timestamp').sort_index()
    return d[~d.index.duplicated()]
pv, dm, gr, bs = rd('*241-Site_DKA_Totals-PV.csv'), rd('*240-Site_DKA_Totals-Site_Demand.csv'), rd('*242-Site_DKA_Totals-Grid.csv'), rd('*239-*BESS*.csv')
for n,d in [('pv',pv),('dm',dm),('gr',gr),('bs',bs)]:
    print(n, d.index.min(), d.index.max(), len(d), d.isna().mean().round(3).to_dict())
def hourly(s):
    g=s.resample('h'); m=g.mean(); c=g.count(); return m.where(c>=6)
h=pd.DataFrame({'p_pv':hourly(pv.Active_Power),'p_load':hourly(dm.Active_Power),'p_grid':hourly(gr.Active_Power),
  'p_bess':hourly(bs.Active_Power),'soc':hourly(bs.State_of_Charge),'ghi':hourly(pv.Global_Horizontal_Radiation),
  't_amb':hourly(pv.Weather_Temperature_Celsius),'rh':hourly(pv.Weather_Relative_Humidity),'gti':hourly(pv.Radiation_Global_Tilted)})
h=h.loc['2024-01-20':'2025-09-03 23:00']
prof=h.groupby(h.index.hour)[['p_pv','ghi','p_load']].mean().round(1); print(prof.T.to_string())
print(h.describe().T.round(1)); print('missing', h.isna().mean().round(3).to_dict())
print('balance check load - (pv+grid+bess):', (h.p_load-(h.p_pv+h.p_grid+h.p_bess)).describe().round(1).to_dict())
h.to_pickle(os.path.join(D, 'hourly.pkl'))
w=pd.concat([pd.read_csv(f, skiprows=3, parse_dates=['time']) for f in sorted(glob.glob(os.path.join(D, 'nwp_alice_202*.csv')))])
w.columns=['time','ghi','t','ws10','rh','cc']; w=w.drop_duplicates('time').set_index('time').sort_index()
# Open-Meteo radiation kept as labelled (best lag by correlation)
d=h.copy()
for c in ['ghi','t','rh','cc','ws10']: d['nwp_'+c]=w[c].reindex(d.index).values
d.index.name='time'; d.to_csv(os.path.join(D, 'data.csv'))
print('nwp missing', d.filter(like='nwp').isna().mean().round(3).to_dict())
print('corr nwp_ghi~ghi', d.nwp_ghi.corr(d.ghi).round(3), ' shifted +1', d.nwp_ghi.shift(1).corr(d.ghi).round(3), ' -1', d.nwp_ghi.shift(-1).corr(d.ghi).round(3))
print('corr nwp_t~t', d.nwp_t.corr(d.t_amb).round(3))
