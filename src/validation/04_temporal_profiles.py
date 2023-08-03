"""
Script to reproduce the subplots shown in Figure 8 in the paper
with the temporal trajectories of reconstructed and in-situ measured
GLAI values per parcel.

.. code-block:: shell

    python 04_temporal_profiles.py

@author: Lukas Valentin Graf
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from eodal.core.scene import SceneCollection
from pathlib import Path

models = ['non_linear', 'sigmoid', 'asymptotic', 'WangEngels']
bbch_range = (30, 59)
black_listed_parcels = ['Bramenwies']

plt.style.use('bmh')
warnings.filterwarnings('ignore')

# define colors for plotting
# use colors from the colorblind palette
model_colors = {
    'sigmoid': '#176d9c',
    'non_linear': '#c38820',
    'asymptotic': '#158b6a',
    'WangEngels': '#ba611b',
    'in-situ': '#99a3ac'
}

model_linestyles = {
    'sigmoid': 'dashed',
    'non_linear': 'solid',
    'asymptotic': 'dotted',
    'WangEngels': 'dashdot'
}
models = ['non_linear', 'asymptotic', 'WangEngels', 'sigmoid', 'in-situ']

model_mappings = {
    'non_linear': 'Non linear',
    'asymptotic': 'Asymptotic',
    'WangEngels': 'Wang Engels',
    'sigmoid': 'Baseline (sigmoid)',
    'in-situ': 'In-situ'
}


def q05(x: pd.Series) -> float:
    """calculate the 5th percentile"""
    return x.quantile(0.05)


def q95(x: pd.Series) -> float:
    """calculate the 95th percentile"""
    return x.quantile(0.95)


def get_data(
        model_output_dir: Path,
        validation_data_dir: Path,
        parcel_data_dir: Path
) -> None:
    """
    Get the model predictions and corresponding in-situ data
    so that a validation can be carried out. The results are
    saved to CSV files in the model output directories so they
    can be used for plotting and error statistics.

    Parameters
    ----------
    model_output_dir : Path
        Path to the directory containing the model output.
    validation_data_dir : Path
        Path to the directory containing the validation data.
    parcel_data_dir : Path
        Path to the directory containing the parcel data.
    """

    # get the validation data
    join_cols = ['date', 'location', 'parcel']
    lai = gpd.read_file(validation_data_dir / 'in-situ_glai.gpkg')
    lai.date = pd.to_datetime(lai.date).dt.tz_localize('Europe/Zurich')
    lai_cols = ['lai', 'geometry'] + join_cols
    bbch = gpd.read_file(validation_data_dir / 'in-situ_bbch.gpkg')
    bbch.date = pd.to_datetime(bbch.date).dt.tz_localize('Europe/Zurich')
    bbch_cols = ['BBCH Rating'] + join_cols
    # join the dataframes on date, location, parcel and point_id
    # in 2023 for the data acquired with the LI-Cor we have to merge
    # on the geometry and the date
    if validation_data_dir.name == '2022':
        val_df = lai[lai_cols].merge(bbch[bbch_cols], on=join_cols,
                                     how='inner')
        val_df.rename(columns={'lai': 'lai_in-situ'}, inplace=True)
    elif validation_data_dir.name == '2023':
        lai['date_only'] = lai.date.dt.date
        bbch['date_only'] = bbch.date.dt.date
        res = []
        for unique_date in lai.date_only.unique():
            lai_date = lai[lai.date_only == unique_date].copy()
            bbch_date = bbch[bbch.date_only == unique_date].copy()
            val_df_date = gpd.sjoin_nearest(
                lai_date, bbch_date)
            columns = ['date_left', 'parcel_left', 'location_left', 'lai',
                       'BBCH Rating', 'geometry']
            val_df_date = val_df_date[columns].copy()
            val_df_date.rename(
                columns={'lai': 'lai_in-situ',
                         'date_left': 'date',
                         'parcel_left': 'parcel',
                         'location_left': 'location'},
                inplace=True)
            res.append(val_df_date)
        val_df = pd.concat(res, axis=0)

    # loop over directories in the model output directory
    for site_dir in model_output_dir.glob('*'):

        # make sure the directory is a directory
        if not site_dir.is_dir():
            continue

        # get the site name from the directory name
        site = site_dir.name.split('_')[1]

        # check site year against year of validation data
        site_year = int(site_dir.name.split('_')[2].split('-')[0])
        year = int(validation_data_dir.name)
        if site_year != year:
            continue

        # read the raw satellite LAI data
        fpath_raw_sat_lai = site_dir.joinpath('raw_lai_values.csv')
        raw_sat_lai = pd.read_csv(fpath_raw_sat_lai)
        # convert to GeoDataFrame
        raw_sat_lai = gpd.GeoDataFrame(
            raw_sat_lai,
            geometry=gpd.GeoSeries.from_xy(
                x=raw_sat_lai.x, y=raw_sat_lai.y, crs='EPSG:32632'))

        # get the corresponnding parcel data (geometry + sowing date)
        parcels_gdf = gpd.read_file(parcel_data_dir / f'{site}.gpkg')
        parcels_gdf['harvest_date'] = pd.to_datetime(
            parcels_gdf['harvest_date'], format='%Y-%m-%d',
            utc=True)
        parcels_gdf['sowing_date'] = pd.to_datetime(
            parcels_gdf['sowing_date'], format='%Y-%m-%d',
            utc=True)
        parcels_gdf = parcels_gdf[
            parcels_gdf.harvest_date.dt.year == year].copy()
        parcels_gdf = parcels_gdf[
            ~parcels_gdf.name.isin(black_listed_parcels)].copy()

        # filter the validation dataframe by site and year
        site_val_df = val_df[val_df.location == site].copy()
        # filter by BBCH range
        site_val_df = site_val_df[
            (site_val_df['BBCH Rating'] >= bbch_range[0]) &
            (site_val_df['BBCH Rating'] <= bbch_range[1])
        ].copy()
        # filter black listed parcels
        site_val_df = site_val_df[
            ~site_val_df.parcel.isin(black_listed_parcels)
        ].copy()
        for model in models:
            model_dir = site_dir.joinpath(model)
            # read the scene collection from pickle (thre could be more than
            # one with different levels of temporal granularity)
            for fpath_scoll in model_dir.glob('*lai.pkl'):
                granularity = fpath_scoll.name.split('_')[0]
                # skip the hourly data for the time being
                if granularity == 'hourly':
                    continue

                # read the scene collection from pickle with the
                # daily LAI values
                scoll = SceneCollection.from_pickle(fpath_scoll)

                # extract the temporal profile of the parcels using some
                # percentiles (5, 25, 50, 75, 95)
                temporal_profile = scoll.get_feature_timeseries(
                    vector_features=parcels_gdf,
                    band_selection=['lai'],
                    method=['percentile_5', 'percentile_25',
                            'percentile_50', 'percentile_75',
                            'percentile_95'])
                # save the temporal profile per parcel and get DAS
                # (days after for year in years:
                parcels_gdf.to_crs(raw_sat_lai.crs, inplace=True)
                for _, parcel_gdf in parcels_gdf.iterrows():

                    # filter the satellite raw LAI by the parcel geometry
                    raw_sat_lai_parcel = raw_sat_lai[
                        raw_sat_lai.geometry.within(parcel_gdf.geometry)
                    ].copy()
                    # group raw lai values by median and 5 and 95 percentile
                    raw_sat_lai_parcel = raw_sat_lai[['lai', 'time']].groupby(
                        'time').agg(['median', q05, q95])
                    raw_sat_lai_parcel = raw_sat_lai_parcel.reset_index()
                    raw_sat_lai_parcel.time = pd.to_datetime(
                        raw_sat_lai_parcel.time, utc=True
                    )
                    raw_sat_lai_parcel['das'] = (
                        raw_sat_lai_parcel.time - parcel_gdf.sowing_date
                    ).dt.days
                    # save to file
                    fname = model_dir.joinpath(
                        f'{parcel_gdf["name"]}_raw_sat-lai.csv')
                    raw_sat_lai_parcel.to_csv(fname)

                    parcel_temporal_profile = temporal_profile[
                        temporal_profile.name == parcel_gdf['name']].copy()
                    parcel_temporal_profile['acquisition_time'] = \
                        pd.to_datetime(
                            parcel_temporal_profile['acquisition_time'],
                            utc=True)
                    parcel_temporal_profile['das'] = (
                        parcel_temporal_profile.acquisition_time -
                        parcel_gdf.sowing_date).dt.days
                    # save the temporal profile for the parcel
                    fname = \
                        model_dir.joinpath(
                            f'{parcel_gdf["name"]}_{granularity}' +
                            '_sat-lai.gpkg')
                    parcel_temporal_profile.to_file(
                        fname, driver='GPKG', index=False)

                    # assign DAS to the validation data
                    site_val_df['das'] = (
                        site_val_df.date - parcel_gdf.sowing_date).dt.days
                    # save the validation data for the parcel
                    parcel_val_df = site_val_df[
                        site_val_df.parcel == parcel_gdf['name']].copy()
                    fname = \
                        model_dir.joinpath(
                            f'{parcel_gdf["name"]}_{granularity}' +
                            '_in-situ.gpkg')
                    parcel_val_df.to_file(
                        fname, driver='GPKG', index=False)
                    print(f'  ->>> Saved {fname}')


def plot_temporal_profiles(
        model_output_dir: Path
) -> None:
    """
    Plot temporal profiles of GLAI trajectories (modeled and in-situ)

    :param model_output_dir: directory with model outputs
    """
    # loop over directories in the model output directory
    for site_dir in model_output_dir.glob('*'):
        if not site_dir.is_dir():
            continue
        res_struct = {}
        parcels = []
        # get the list of subdirectories (models)
        for model_dir in site_dir.glob('*'):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            res_struct[model] = {}
            res_struct['in-situ'] = {}
            res_struct['sat_raw'] = {}
            # get the list of temporal profiles
            for fpath in model_dir.glob('*_sat-lai.gpkg'):
                parcel = fpath.name.split('_')[0]
                parcels.append(parcel)
                if parcel not in res_struct[model].keys():
                    res_struct[model][parcel] = {}
                res_struct[model][parcel] = fpath
                # append the in-situ data
                res_struct['in-situ'][parcel] = \
                    model_dir.joinpath(f'{parcel}_daily_in-situ.gpkg')
                # append the raw data
                res_struct['sat_raw'][parcel] = \
                    model_dir.joinpath(f'{parcel}_raw_sat-lai.csv')

        # loop over the parcels
        for parcel in np.unique(parcels):
            f, ax = plt.subplots(1, 1, figsize=(10, 5))
            for model in models:
                if parcel not in res_struct[model].keys():
                    continue
                # read the temporal profile
                temporal_profile = gpd.read_file(res_struct[model][parcel])
                # read the in-situ data
                in_situ = gpd.read_file(res_struct['in-situ'][parcel])
                # plot the temporal profile

                if model not in ['in-situ', 'sat_raw']:
                    ax.plot(
                        temporal_profile.das,
                        temporal_profile.percentile_50,
                        label=model_mappings[model],
                        color=model_colors[model],
                        linestyle=model_linestyles[model])
                    ax.fill_between(
                        temporal_profile.das,
                        temporal_profile.percentile_5,
                        temporal_profile.percentile_95,
                        alpha=0.1,
                        color=model_colors[model])

                # plot the in-situ data
                elif model == 'in-situ':
                    ax.scatter(
                        in_situ.das,
                        in_situ['lai_in-situ'],
                        label='in-situ')
            ax.set_xlabel('Days After Sowing', fontsize=18)
            ax.set_ylabel(r'GLAI [m$^2$ m$^{-2}$]', fontsize=18)
            ax.set_ylim(0, 8)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            site_name = site_dir.name.replace("_", " ").split('farm ')[1]
            ax.set_title(f'{site_name} - {parcel}', fontsize=18)
            ax.legend(fontsize=16)
            fname_plot = site_dir.joinpath(f'{parcel}_temporal_profile.png')
            f.savefig(fname_plot, dpi=300, bbox_inches='tight')
            plt.close(f)

            # plot raw LAI data
            f, ax = plt.subplots(1, 1, figsize=(10, 5))
            temporal_profile = pd.read_csv(res_struct['sat_raw'][parcel],
                                           header=1, index_col=0)
            temporal_profile.columns = ['time', 'median', 'q05', 'q95', 'das']
            ax.plot(
                temporal_profile.das,
                temporal_profile['median'],
                label='Raw Satellite LAI',
                marker='x',
                color='blue'
            )
            ax.fill_between(
                temporal_profile.das,
                temporal_profile['q05'],
                temporal_profile['q95'],
                alpha=0.5,
                label='5-95 Percentile Spread')
            ax.legend()
            ax.set_xlabel('Days After Sowing', fontsize=18)
            ax.set_ylabel(r'GLAI [m$^2$ m$^{-2}$]', fontsize=18)
            ax.set_ylim(0, 8)
            ax.set_title(f'{site_dir.name.replace("_", " ")} - {parcel}')
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            fname_plt_raw = site_dir.joinpath(
                f'{parcel}_temporal_profile_raw.png')
            f.savefig(fname_plt_raw, dpi=300, bbox_inches='tight')
            plt.close(f)

            print(f'  ->>> Saved {fname_plot}')


if __name__ == '__main__':

    model_output_dir = Path('results/validation_sites')

    # in-situ validation data
    years = [2022, 2023]

    # go through the years and extract the data. This is done
    # only once. The results are saved to CSV files in the
    # model output directories so they can be used for plotting
    # and error statistics.

    for year in years:
        validation_data_dir = Path('data/in-situ') / str(year)
        parcel_data_dir = Path('data/Test_Sites')
        get_data(
            model_output_dir=model_output_dir,
            validation_data_dir=validation_data_dir,
            parcel_data_dir=parcel_data_dir)

    # plot the temporal profiles
    plot_temporal_profiles(model_output_dir=model_output_dir)
