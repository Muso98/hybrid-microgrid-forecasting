import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kelmarsh")
import pandas as pd, numpy as np, json
s = pd.read_csv(os.path.join(D, 'kelmarsh_scada_2024_hourly.csv'))
s['time'] = pd.to_datetime(s.hour + ':00')
w = pd.read_csv(os.path.join(D, 'nwp_kelmarsh.csv'), skiprows=3, parse_dates=['time'])
w.columns = ['time','ghi','t','ws10','ws80','ws100','wd100','rh','cc']
for c in ['ws10','ws80','ws100']: w[c] = w[c]/3.6
w['ghi'] = w.ghi.shift(-1)          # hour-ending -> hour-beginning
w = w.set_index('time')
idx = pd.date_range('2024-01-21 00:00','2024-12-31 22:00',freq='h')
piv = {c: s.pivot(index='time', columns='turbine', values=c).reindex(idx) for c in ['ws','wd','p','tamb','rpm','pitch','avail']}
d = pd.DataFrame(index=idx)
for k in range(1,7):
    for c in ['ws','p','pitch']: d[f'{c}{k}'] = piv[c][k]
d['ws_mean'] = piv['ws'].mean(1); d['ws_std'] = piv['ws'].std(1)
wd = np.radians(piv['wd']); d['wd_s'] = np.sin(wd).mean(1); d['wd_c'] = np.cos(wd).mean(1)
d['tamb'] = piv['tamb'].mean(1); d['rpm'] = piv['rpm'].mean(1); d['avail'] = piv['avail'].mean(1)
d['p_farm'] = piv['p'].sum(1, min_count=6)/1000.0     # MW, NaN if any turbine missing
for c in ['ghi','t','ws10','ws100','wd100','rh']: d['nwp_'+c] = w[c].reindex(idx).values
d['nwp_wd_s'] = np.sin(np.radians(d.nwp_wd100)); d['nwp_wd_c'] = np.cos(np.radians(d.nwp_wd100))
d.index.name = 'time'; d.to_csv(os.path.join(D, 'data.csv'))
print(d.shape); print(d.isna().sum()[d.isna().sum()>0])
print('corr nwp ws100 vs ws_mean', d.nwp_ws100.corr(d.ws_mean).round(3), 'p_farm mean', d.p_farm.mean().round(3), 'cap 12.3')
print('frac hours avail<1:', (d.avail<0.999).mean().round(3), ' p_farm<0.05 while ws_mean>6:', ((d.p_farm<0.05)&(d.ws_mean>6)).mean().round(4))
