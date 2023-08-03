"""
Combine the GLAI data over the validation sites covering the period for which
in-situ ratings are available (BBCH 30 to 59). This script brings the data
into "raw" GLAI trajectories per pixel so that the temperature response can be
applied in the next step.

Usage:

.. code-block:: shell

    python 03_generate_raq_s2_trait_trajectories.py

@author: Lukas Valentin Graf
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eodal.config import get_settings
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from pathlib import Path
from typing import List

# logging and plotting settings
logger = get_settings().logger
plt.style.use('bmh')

# datetime formats of the meteo data
formats = {
    'Strickhof': '%d.%m.%Y %H:%M',
    'SwissFutureFarm': '%Y-%m-%d %H:%M:%S'
}
# BBCH range to consider
BBCH_MIN = 25
BBCH_MAX = 59
BBCH_TOLERANCE = 7  # days


def get_lai_model_results_for_validation(
        trait_dir: Path,
        insitu_dir: Path,
        test_site_dir: Path,
        years: List[int]
) -> None:
    """
    Extract LAI model results for validation

    :param trait_dir:
        directory with S2 model results
    :param insitu_dir:
        directory with in-situ data
    :param test_site_dir:
        directory with test site data
    :param years:
        years with in-situ data
    """
    # loop over years
    for year in years:

        # the in-situ data is stored by year
        insitu_year_dir = insitu_dir.joinpath(f'{year}')
        # read the data into a GeoDataFrame
        fpath_bbch = insitu_year_dir.joinpath('in-situ_bbch.gpkg')
        insitu = gpd.read_file(fpath_bbch)
        # convert date to pd.to_datetime
        insitu['date'] = pd.to_datetime(insitu['date'])

        # filter for stem elongation phase
        insitu = insitu[
            (insitu['BBCH Rating'] >= BBCH_MIN) &
            (insitu['BBCH Rating'] <= BBCH_MAX)].copy()

        # loop over farms in in-situ data
        for farm in insitu['location'].unique():
            # get the farm directory
            farm_dir = trait_dir.joinpath(farm)
            if not farm_dir.exists():
                continue

            # get parcel geometries for the farm
            fpath_gpkg = test_site_dir.joinpath(f'{farm}.gpkg')
            test_site_gdf = gpd.read_file(fpath_gpkg)
            # get only those parcels whose growth period is in the
            # range of the insitu data. The sowing date must be in
            # the year before the in-situ data was taken.
            test_site_gdf['sowing_year'] = pd.to_datetime(
                test_site_gdf['sowing_date']).dt.year
            test_site_gdf = test_site_gdf[
                test_site_gdf['sowing_year'] == year - 1].copy()

            # get the in-situ BBCH data for the farm
            farm_insitu = insitu[insitu['location'] == farm].copy()
            farm_insitu_min = farm_insitu['date'].min() - \
                pd.Timedelta(days=BBCH_TOLERANCE)
            farm_insitu_max = farm_insitu['date'].max() + \
                pd.Timedelta(days=BBCH_TOLERANCE)

            # loop over scenes in farm
            for scene_dir in farm_dir.glob('*.SAFE'):
                # get the date of the scene
                date = pd.to_datetime(scene_dir.name.split('_')[2])
                # check if the scene is in the stem elongation phase
                if not (date >= farm_insitu_min and
                        date <= farm_insitu_max):
                    continue
                # check if an inversion result for the stem-elongation phase
                # has been generated already
                # if not, we skip the scene
                # if the date is within the tolerance window, we might
                # use the output of the previous phenological phase
                if date - farm_insitu_min < pd.Timedelta(days=BBCH_TOLERANCE):
                    fpath_traits = scene_dir.joinpath(
                        'germination-endoftillering_lutinv_traits.tiff')
                elif farm_insitu_max - date < pd.Timedelta(days=BBCH_TOLERANCE):  # noqa: E501
                    fpath_traits = scene_dir.joinpath(
                        'flowering-fruitdevelopment-plantdead_lutinv_traits.tiff')  # noqa: E501
                else:
                    fpath_traits = scene_dir.joinpath(
                        'stemelongation-endofheading_lutinv_traits.tiff')

                if not fpath_traits.exists():
                    continue

                # read the trait data for the parcel geometry using eodal
                ds = RasterCollection.from_multi_band_raster(
                    fpath_raster=fpath_traits, vector_features=test_site_gdf)
                # save the result in a validation sub-folder
                fpath_validation = scene_dir.joinpath('validation')
                fpath_validation.mkdir(exist_ok=True)
                # save ds to the validation folder as a geoTiff
                ds.to_rasterio(fpath_validation.joinpath('traits.tiff'))

                logger.info(f'Saved traits for {farm}: {scene_dir.name}')


def extract_raw_lai_timeseries(
        test_sites_dir: Path,
        s2_trait_dir: Path,
        meteo_dir: Path,
        out_dir: Path,
        farms: list[str],
        years: list[int],
        traits: list[str] = ['lai', 'lai_q05', 'lai_q95']
) -> None:
    """
    Extract the raw LAI timeseries for the validation sites.

    :param test_sites_dir:
        Path to the directory containing the test sites.
    :param s2_trait_dir:
        Path to the directory containing the Sentinel-2 trait data.
    :param meteo_dir:
        Path to the directory containing the meteorological data.
    :param out_dir:
        Path to the directory where the results should be stored.
    :param farms:
        List of farms to be considered.
    :param years:
        List of years to be considered.
    :param traits:
        List of traits to be considered.
    """
    # loop over years
    for year in years:
        # loop over farms
        for farm in farms:
            # go through the scenes and only consider those that
            # are in the current year and have a subfolder called
            # 'validation'
            scene_dir_farm = s2_trait_dir.joinpath(farm)
            # open an empty SceneCollection
            scoll = SceneCollection()
            for scene in scene_dir_farm.glob('*.SAFE'):
                # get the date of the scene
                scene_date = pd.to_datetime(
                    scene.name.split('_')[2]).tz_localize(
                        'Europe/Zurich')
                if scene_date.year != year:
                    continue
                # check for a subfolder called 'validation'
                if not scene.joinpath('validation').exists():
                    continue
                # get the LAI data
                lai_file = scene.joinpath('validation', 'traits.tiff')
                # read results into RasterCollection
                s2_traits = RasterCollection.from_multi_band_raster(
                    fpath_raster=lai_file)
                # check if the result contains actual data
                if np.isnan(s2_traits['lai'].values).all():
                    continue
                scene_props = SceneProperties(
                    acquisition_time=scene_date,
                    product_uri=scene.name
                )
                s2_traits.scene_properties = scene_props

                scoll.add_scene(s2_traits)

            # make sure the scene are sorted chronologically
            scoll = scoll.sort('asc')
            # continue if no scenes were found
            if scoll.empty:
                logger.warn(
                    f'No scenes found for {farm} in {year}.')
                continue

            # extract the meteorological data (hourly)
            fpath_meteo_site = meteo_dir.joinpath(
                f'{farm}_Meteo_hourly.csv')
            meteo_site = pd.read_csv(fpath_meteo_site)
            meteo_site.time = pd.to_datetime(
                meteo_site.time, format=formats[farm])
            meteo_site.index = meteo_site.time
            # we only need to have meteorological data for the S2 observations
            # selected
            try:
                min_time = pd.to_datetime(scoll.timestamps[0].split('+')[0])
            except Exception as e:
                logger.error(e)
                continue
            max_time = pd.to_datetime(scoll.timestamps[-1].split('+')[0]) + \
                pd.Timedelta(days=1)   # add one day to include the last day
            meteo_site_parcel = meteo_site[
                min_time.date():max_time.date()].copy()[['time', 'T_mean']]
            meteo_site_parcel.index = [
                x for x in range(meteo_site_parcel.shape[0])]

            # save data to output directory
            out_dir_parcel = out_dir.joinpath(
                f'farm_{farm}_{min_time.date()}-{max_time.date()}')
            out_dir_parcel.mkdir(exist_ok=True)

            # save "raw" LAI values as pickle
            fname_raw_lai = out_dir_parcel.joinpath('raw_lai_values.pkl')
            with open(fname_raw_lai, 'wb+') as dst:
                dst.write(scoll.to_pickle())

            # save "raw" LAI values as table using all pixels in the parcels
            # so that it is easier to work with the data
            # we use xarray as an intermediate to convert it to a pandas
            # DataFrame
            xarr = scoll.to_xarray()
            for trait in traits:
                df = xarr.to_dataframe(name=trait).reset_index()
                df = df[df.band == trait].copy()
                # drop nan's (these are the pixels outside the parcel)
                df.dropna(inplace=True)
                # drop the band name column since it is redundant
                df.drop('band', axis=1, inplace=True)
                # save the DataFrame as CSV file
                fname_csv = out_dir_parcel.joinpath(f'raw_{trait}_values.csv')
                df.to_csv(fname_csv, index=False)

            # plot data LAI as a map
            f = scoll.plot(
                band_selection='lai',
                figsize=(5*len(scoll), 5),
                max_scenes_in_row=len(scoll)
            )
            f.savefig(out_dir_parcel.joinpath('raw_lai_values.png'))
            plt.close(f)

            # save hourly meteo data
            meteo_site_parcel.to_csv(out_dir_parcel.joinpath(
                'hourly_mean_temperature.csv'), index=False)
            f, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                meteo_site_parcel.time, meteo_site_parcel['T_mean'].astype(
                    float))
            ax.set_ylabel('Mean Air Temperature 2m above ground [deg C]')
            plt.xticks(rotation=90)
            f.savefig(out_dir_parcel.joinpath(
                'hourly_mean_temperature.png'), bbox_inches='tight')
            plt.close(f)

            logger.info(
                f'Prepared LAI data for parcel {farm} in {year}')


if __name__ == '__main__':

    # define paths
    test_sites_dir = Path('data/Test_Sites')
    s2_trait_dir = Path('data/S2_Traits')
    meteo_dir = Path('data/Meteo')
    insitu_dir = Path('data/in-situ')

    # make output directory
    out_dir = Path('results/validation_sites')
    out_dir.mkdir(exist_ok=True)

    # sites and years to consider
    farms = ['Strickhof', 'SwissFutureFarm']
    years = [2022, 2023]

    # step 1: get the S2 GLAI observations in the BBCH range of interest
    get_lai_model_results_for_validation(
        trait_dir=trait_dir,
        insitu_dir=insitu_dir,
        years=years,
        test_site_dir=test_sites_dir)

    # step 2: extract the GLAI trajectory per parcel for all pixels.
    extract_raw_lai_timeseries(
        test_sites_dir=test_sites_dir,
        s2_trait_dir=s2_trait_dir,
        meteo_dir=meteo_dir,
        out_dir=out_dir,
        farms=farms,
        years=years
    )
