
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('bmh')
plt.rcParams['font.size'] = 18


path_lai = Path('/mnt/ides/Lukas/software/sentinel2_crop_traits/data/in_situ_traits_2022/in-situ_glai.gpkg')
path_bbch = Path('/mnt/ides/Lukas/software/sentinel2_crop_traits/data/in_situ_traits_2022/in-situ_bbch.gpkg')

lai = gpd.read_file(path_lai)
bbch = gpd.read_file(path_bbch)

# filter lai and bbch by location "Areneberg"
lai = lai[lai['location'] == 'Arenenberg'].copy()
bbch = bbch[bbch['location'] == 'Arenenberg'].copy()

# merge lai and bbch on date and point id. convert the
# dates to datetime objects before
lai['date'] = pd.to_datetime(lai['date'])
bbch['date'] = pd.to_datetime(bbch['date'])

df = lai.merge(bbch, on=['date', 'point_id'], suffixes=('_lai', '_bbch'))
df = df[['date', 'point_id', 'lai', 'BBCH Rating']].copy()
df.drop_duplicates(inplace=True)
df = df[(df['BBCH Rating'] >= 30) & (df['BBCH Rating'] <= 59)].copy()
df.rename(columns={'BBCH Rating': 'BBCH', 'point_id': 'Point_ID', 'lai': 'LAI_value'}, inplace=True)
df.to_csv('results/dose_reponse_in-situ/CH_Arenenberg/LAI_ARB_Raw-Data.csv', index=False)

# plot the data as time series by point id
f, ax = plt.subplots(figsize=(10, 5))
df.groupby('Point_ID').plot(x='date', y='LAI_value', legend=False, ax=ax)
ax.set_ylabel(r'In-situ Green Leaf Area Index [m$^2$ $m^{-2}$]')
ax.set_xlabel('Time')
f.savefig('results/dose_reponse_in-situ/CH_Arenenberg/LAI_ARB_Raw-Data_plot.png', dpi=300, bbox_inches='tight')
plt.close(f)

# prepare the meteo data
path_meteo = Path('results/dose_reponse_in-situ/CH_Arenenberg/order111600/order_111600_data.txt')
meteo = pd.read_csv(path_meteo, sep=';')
meteo['time'] = pd.to_datetime(meteo['time'], format='%Y%m%d%H')
meteo.rename(columns={'tre200h0': 'T_mean'}, inplace=True)
meteo[['time', 'T_mean']].to_csv('results/dose_reponse_in-situ/CH_Arenenberg/Meteo_ARB.csv', index=False)
meteo