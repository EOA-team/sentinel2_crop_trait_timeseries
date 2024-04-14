'''
Apply the proposed DRC approach to reconstruct GLAI time series
from raw satellite observation trajectories and air temperature data.

Usage:

.. code-block:: shell

    python 04_reconstruct_s2_traits.py

@author: Flavian Tschurr and Lukas Valentin Graf
'''

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

from eodal.config import get_settings
from eodal.core.band import Band, GeoInfo
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from functools import reduce
from pathlib import Path
from typing import List

from ensemble_kalman_filter import EnsembleKalmanFilter
from temperature_response import Response

logger = get_settings().logger
warnings.filterwarnings('ignore')
plt.style.use('bmh')

# set seed to make results reproducible
np.random.seed(42)

# noise level for temperature data
noise_level = 5  # in percent
# uncertainty in LAI data (relative)
lai_uncertainty = 5  # in percent


def plot_interpolated_lai(
        model_sims_between_points: pd.DataFrame,
        enskf: EnsembleKalmanFilter
) -> plt.Figure:
    """
    Plot the interpolated LAI time series after data assimilation.

    Parameters
    ----------
    model_sims_between_points : pd.DataFrame
        Data frame containing the interpolated LAI time series.
    enskf : EnsembleKalmanFilter
        Ensemble Kalman Filter object.
    return : plt.Figure
        Figure containing the plot.
    """
    f, ax = plt.subplots(ncols=1, nrows=3, figsize=(10, 8),
                         sharex=True, sharey=False)

    # plot the temperature time series
    ax[0].plot(
        model_sims_between_points['time'],
        model_sims_between_points['T_mean'],
    )
    ax[0].set_xlabel('')
    ax[0].set_ylabel(r'Air Temperature [$^\circ$C]')

    # plot the assimilation
    enskf.plot_new_states(ax=ax[1])
    ax[1].set_xlabel('')
    ax[1].set_ylim(0, 8)
    ax[1].set_ylabel(r'GLAI [$m^2$ $m^{-2}$]')

    # plot the interpolated LAI time series
    ax[2].plot(
        model_sims_between_points['time'],
        model_sims_between_points['lai'],
        color='red',
        label='Satellite GLAI',
        marker='o'
    )
    ax[2].plot(
        model_sims_between_points['time'],
        model_sims_between_points['reconstructed_lai_mean'],
        color='blue',
        label='Reconstructed GLAI'
    )
    ax[2].fill_between(
        model_sims_between_points['time'],
        model_sims_between_points['reconstructed_lai_mean'] -
        model_sims_between_points['reconstructed_lai_std'],
        model_sims_between_points['reconstructed_lai_mean'] +
        model_sims_between_points['reconstructed_lai_std'],
        color='blue',
        alpha=0.2,
        label='Uncertainty'
    )
    ax[2].set_xlabel('Time')
    ax[2].set_ylim(0, 8)
    # rotate x labels by 45 degrees
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)
    ax[2].set_ylabel(r'GLAI [$m^2$ $m^{-2}$]')
    ax[2].legend()
    return f


def prepare_lai_ts(
        lai_pixel_ts: pd.Series,
        percentage_datapoints_to_remove: float = 0.1
) -> pd.Series:
    """
    Prepare LAI time series for the temperature response function.

    Parameters
    ----------
    lai_pixel_ts : pd.Series
        LAI time series.
    percentage_datapoints_to_remove : float, optional
        Percentage of data points to remove. Default is 0.1.
    return : pd.Series
        Prepared LAI time series.
    """
    lai_pixel_ts.sort_values(by='time', inplace=True)
    lai_pixel_ts.index = [x for x in range(len(lai_pixel_ts))]

    # randomly remove x percent of the data
    indices_to_remove = np.random.choice(
        lai_pixel_ts.index,
        int(len(lai_pixel_ts) * percentage_datapoints_to_remove),
        replace=False)
    lai_pixel_ts.loc[indices_to_remove, 'lai'] = np.nan

    # apply a simple outlier filtering
    # values smaller than one standard deviation are removed
    # we look in negative direction, only
    # the exception is the first value
    lai_values = lai_pixel_ts['lai'].values.copy()
    mean, std = np.mean(lai_values), np.std(lai_values)
    lai_values[1:] = np.where(
        lai_values[1:] < mean - std,
        np.nan,
        lai_values[1:]
    )
    # get indices of nan values
    nan_indices = np.argwhere(np.isnan(lai_values)).flatten()
    # remove nan values from lai_pixel_ts
    lai_pixel_ts = lai_pixel_ts[
        ~lai_pixel_ts.index.isin(nan_indices)].copy()

    return lai_pixel_ts


def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def interpolate_between_assimilated_points(
        measurement_index: List[int],
        meteo_pixel: pd.DataFrame,
        response: Response
) -> pd.DataFrame:
    """
    Interpolate assimilated LAI values between satellite observations.

    Parameters
    ----------
    measurement_index : List[int]
        List of measurement indices.
    meteo_pixel : pd.DataFrame
        Meteo data.
    response : Response
        Response object.

    Returns
    -------
    pd.DataFrame
        Interpolated LAI values.
    """
    model_sims_between_points = []
    # loop over measurement points
    for i in range(len(measurement_index)-1):
        meteo_time_window = meteo_pixel.loc[
            measurement_index[i]:measurement_index[(i+1)]].copy()
        # calculate the temperature response
        meteo_time_window['temp_response'] = \
            response.get_response(
                meteo_time_window['T_mean'])
        # get cumulative sum of temperature response
        meteo_time_window['temp_response_cumsum'] = \
            meteo_time_window['temp_response'].cumsum()
        # scale values between lai_value_start and lai_value_end
        in_min = meteo_time_window['temp_response_cumsum'].iloc[0]
        in_max = meteo_time_window['temp_response_cumsum'].iloc[-1]

        for measure in ['mean', 'std']:
            out_min = meteo_time_window[
                f'reconstructed_lai_{measure}'].iloc[0]
            out_max = meteo_time_window[
                f'reconstructed_lai_{measure}'].iloc[-1]
            meteo_time_window[f'reconstructed_lai_{measure}'] = \
                meteo_time_window['temp_response_cumsum'].apply(
                    lambda x: rescale(x, in_min, in_max, out_min, out_max))

        model_sims_between_points.append(meteo_time_window)

    model_sims_between_points = pd.concat(
        model_sims_between_points, axis=0)
    return model_sims_between_points


def merge_with_meteo(
        meteo: pd.DataFrame,
        lai_pixel_ts: pd.DataFrame,
        covariate_granularity: str
) -> pd.DataFrame:
    """
    Merge meteo data with LAI time series.

    Parameters
    ----------
    meteo : pd.DataFrame
        Meteo data.
    lai_pixel_ts : pd.DataFrame
        LAI time series.
    covariate_granularity : str
        Granularity of the covariates. Either 'daily' or 'hourly'.
    Returns
    -------
    pd.DataFrame
        Merged data.
    """
    _meteo = meteo.copy()
    if covariate_granularity == 'daily':
        # merge on the data
        _meteo['date'] = _meteo['time'].dt.date
        lai_pixel_ts['date'] = lai_pixel_ts['time'].dt.date
        meteo_pixel = pd.merge(
            _meteo, lai_pixel_ts, on='date', how='left')
        meteo_pixel['time'] = meteo_pixel['date']
        cols_to_drop = [x for x in meteo_pixel.columns
                        if x.endswith('_y') or x.endswith('_x')]
        meteo_pixel = meteo_pixel.drop(cols_to_drop, axis=1)
    else:
        meteo_pixel = pd.merge(
            _meteo, lai_pixel_ts, on='time', how='left')
    return meteo_pixel


def apply_temperature_response(
        parcel_lai_dir: Path,
        dose_response_parameters: Path,
        response_curve_type,
        covariate_granularity,
        percentage_datapoints_to_remove: float = 0.1,
        n_sim: int = 50,
        n_plots: int = 20
) -> None:
    """
    Apply the temperature response function to the LAI time series.

    Parameters
    ----------
    parcel_lai_dir : Path
        Path to the directory containing the LAI time series.
    dose_response_parameters : Path
        Path to the dose response parameters.
    response_curve_type : str
        Type of the response curve.
    covariate_granularity : str
        Granularity of the covariate.
    percentage_datapoints_to_remove : float, optional
        Percentage of data points to remove. Default is 0.1.
    n_sim : int, optional
        Number of simulations for the ensemble Kalman filter.
        Default is 50.
    n_plots : int, optional
        Number of plots to generate (random selection). Default is 20.
    """
    # read in dose response paramters
    path_paramters = Path.joinpath(
        dose_response_parameters,
        response_curve_type,
        f'{response_curve_type}_granularity_{covariate_granularity}' +
        '_parameter_T_mean.csv')

    params = pd.read_csv(path_paramters)
    params = dict(zip(params['parameter_name'], params['parameter_value']))

    # loop over parcels and read the data
    for parcel_dir in parcel_lai_dir.glob('*'):

        if not parcel_dir.is_dir():
            continue

        if parcel_dir.name == 'error_stats_plots':
            continue

        logger.info(
            f'Working on {parcel_dir.name} to get ' +
            f'{covariate_granularity} LAI values ' +
            f'using {response_curve_type} response curve')

        # for the test pixels we can use our phenology model
        fpath_relevant_phase = parcel_dir.joinpath('relevant_phase.txt')
        if fpath_relevant_phase.exists():
            with open(fpath_relevant_phase, 'r') as src:
                phase = src.read()
            if phase != 'stemelongation-endofheading':
                continue

        # make an output dir
        output_dir = parcel_dir.joinpath(response_curve_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_dir_plots = output_dir.joinpath('plots')
        output_dir_plots.mkdir(parents=True, exist_ok=True)

        # leaf area index data
        # We read the actual inversion result (raw_lai_values.csv) and
        # the upper and lower bound of LAI quantiles (q_05 and q_95)
        # which give us a measure of uncertainty of the inversion.
        temp_lai_list = []
        for fpath_lai in parcel_dir.glob('raw_lai*_values.csv'):
            temp_lai = pd.read_csv(fpath_lai)
            # we need to ensure that coordinates are rounded to whole digits
            # otherwise we will get problems when identifying the single pixels
            # in the LAI data
            temp_lai['x'] = temp_lai['x'].round(0)
            temp_lai['y'] = temp_lai['y'].round(0)
            temp_lai['time'] = pd.to_datetime(
                temp_lai['time'], format='ISO8601', utc=True).dt.floor('H')
            # convert to GeoDataFrame to make sure the coordinates are
            # correctly interpreted (all coordinates are UTM zone 32N)
            temp_lai = gpd.GeoDataFrame(
                temp_lai,
                geometry=gpd.points_from_xy(temp_lai.x, temp_lai.y),
                crs='EPSG:32632'
            )
            temp_lai_list.append(temp_lai)

        # merge the LAI data
        lai = reduce(
            lambda left, right: pd.merge(
                left, right, on=['time', 'x', 'y'], how='outer'),
            temp_lai_list)
        # delete columns ending with _y and _x
        cols_to_drop = [x for x in lai.columns
                        if x.endswith('_y') or x.endswith('_x')]
        lai = lai.drop(cols_to_drop, axis=1)

        # some geometries are shifted by 1 m in x and y direction
        # we need to correct this (still a bug in eodal 0.2.1)
        geoms = lai.geometry
        geoms_corrected_indices = []
        for unique_geom in geoms:
            # continue if the geometry is already processed
            if unique_geom in geoms_corrected_indices:
                continue
            distance = abs(unique_geom.distance(lai.geometry).sort_values())
            distance = distance[distance < 5].copy()
            if distance.empty:
                continue
            # get the index of the geometry which is less than 5 m apart
            # from the unique geometry
            close_geom_indices = distance.index
            for close_geom_index in close_geom_indices:
                lai.loc[close_geom_index, 'geometry'] = unique_geom
                lai.loc[close_geom_index, 'x'] = unique_geom.x
                lai.loc[close_geom_index, 'y'] = unique_geom.y
                geoms_corrected_indices.append(close_geom_index)

        # meteorological data
        fpath_meteo = parcel_dir.joinpath('hourly_mean_temperature.csv')
        meteo = pd.read_csv(fpath_meteo)
        # check for implausible values (lower than -30 degrees or
        # higher than 50 degrees Celsius)
        meteo['plausible'] = meteo['T_mean'].apply(
            lambda x: False if x < -30 or x > 50 else True)
        if not meteo.plausible.all():
            logger.error(
                f'Implausible values in {fpath_meteo}. Skipping...')
            continue
        # ensure timestamp format
        meteo['time'] = pd.to_datetime(
            meteo['time'], utc=True).dt.floor('H')
        # sort by time
        meteo = meteo.sort_values(by='time')

        # if the granulatiry of the covariate is daily, we need to
        # resample the meteo data
        if covariate_granularity == 'daily':
            meteo = meteo.resample('D', on='time').mean().reset_index()

        # calculate temperature response and write into
        # the meteo df
        Response_calculator = Response(
            response_curve_type=response_curve_type,
            response_curve_parameters=params)

        # loop over pixels
        interpolated_pixel_results = []
        # determine randomly for which pixel_coords we want to
        # generate plots
        try:
            pixel_coords_to_plot = random.sample(
                list(lai.groupby(['y', 'x']).groups.keys()), n_plots)
        except ValueError:
            pixel_coords_to_plot = random.sample(
                list(lai.groupby(['y', 'x']).groups.keys()), 1)

        for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):

            plot_pixel = pixel_coords in pixel_coords_to_plot

            lai_pixel_ts = prepare_lai_ts(
                lai_pixel_ts=lai_pixel_ts,
                percentage_datapoints_to_remove=percentage_datapoints_to_remove
            )
            # special case: if we only have a single measurement
            # we cannot interpolate between the assimilated points.
            if len(lai_pixel_ts) == 1:
                continue

            meteo_pixel = merge_with_meteo(
                meteo=meteo,
                lai_pixel_ts=lai_pixel_ts,
                covariate_granularity=covariate_granularity)

            # STEP 1: Data Assimilation using Ensemble Kalman Filter
            # setup Ensemble Kalman Filter
            enskf = EnsembleKalmanFilter(
                state_vector=meteo_pixel,
                response=Response_calculator,
                n_sim=n_sim)
            # run the filter to assimilate data
            enskf.run()

            # STEP 2: Interpolate between the assimilated points
            # get assimilated results at the measurement values
            # and interpolate between them using scaled temperature
            # response to get a continuous LAI time series without
            # breaks resulting from the assimilation
            measurement_indices = meteo_pixel[
                meteo_pixel['lai'].notnull()]['time'].tolist()

            meteo_pixel['reconstructed_lai_mean'] = np.nan
            meteo_pixel['reconstructed_lai_std'] = np.nan
            meteo_pixel['reconstructed_lai_diff'] = np.nan
            meteo_pixel.index = meteo_pixel['time']

            # STEP 3: get the assimilated LAI values
            # ignore the last element as we do not have an uncertainty
            # estimate for it
            for i in range(len(measurement_indices)):
                measurement_index = measurement_indices[i]
                assimilated_lai_values = \
                    enskf.new_states.loc[measurement_index].iloc[-1]
                # get mean and standard deviation of the ensemble
                # at the measurement point for which an S2 observation
                # is available
                assimilated_lai_value_mean = \
                    np.mean(assimilated_lai_values)
                assimilated_lai_value_std = \
                    np.std(assimilated_lai_values)
                meteo_pixel.loc[measurement_index,
                                'reconstructed_lai_mean'] = \
                    assimilated_lai_value_mean
                meteo_pixel.loc[measurement_index,
                                'reconstructed_lai_std'] = \
                    assimilated_lai_value_std
                # calculate the difference between the assimilated
                # LAI values (i.e., the slope between the assimilated
                # points)
                # the exception is the first and last measurement
                # point for which we set the difference to 0
                if i == 0:
                    meteo_pixel.loc[measurement_index,
                                    'reconstructed_lai_diff'] = 0
                elif i == len(measurement_indices) - 1:
                    meteo_pixel.loc[measurement_index,
                                    'reconstructed_lai_diff'] = 0
                else:
                    previous_measurement_index = measurement_indices[i-1]
                    meteo_pixel.loc[measurement_index,
                                    'reconstructed_lai_diff'] = \
                        assimilated_lai_value_mean - \
                        meteo_pixel.loc[previous_measurement_index,
                                        'reconstructed_lai_mean']

            # STEP 4: interpolate between the assimilated points.
            # Set the measurement indices so that only data points are
            # considered for interpolation that do not cause a drop
            # in LAI (i.e., the reconstructed_lai_diff) must not be
            # negative
            measurement_indices = meteo_pixel[
                (meteo_pixel['lai'].notnull()) &
                (meteo_pixel['reconstructed_lai_diff'] >= 0)]['time'].tolist()

            # interpolate between the assimilated points
            # using the scaled temperature response
            try:
                model_sims_between_points = \
                    interpolate_between_assimilated_points(
                        measurement_index=measurement_indices,
                        meteo_pixel=meteo_pixel,
                        response=Response_calculator)
            except ValueError as e:
                logger.error(
                    f'{parcel_dir.name} {pixel_coords} failed: {e}')
                continue

            # plot time series
            if plot_pixel:
                f = plot_interpolated_lai(model_sims_between_points, enskf)
                f.savefig(
                    output_dir_plots.joinpath(
                        f'interpolated_lai_{pixel_coords[0]}'
                        f'_{pixel_coords[1]}_{covariate_granularity}.png'),
                    dpi=300, bbox_inches='tight')
                plt.close(f)

            # save results to DataFrame
            lai_interpolated_df = pd.DataFrame({
                'time': model_sims_between_points['time'],
                'lai': model_sims_between_points[
                    'reconstructed_lai_mean'],
                'lai_minus_std': model_sims_between_points[
                    'reconstructed_lai_mean'] - model_sims_between_points[
                        'reconstructed_lai_std'],
                'lai_plus_std': model_sims_between_points[
                    'reconstructed_lai_mean'] + model_sims_between_points[
                        'reconstructed_lai_std'],
                'y': pixel_coords[0],
                'x': pixel_coords[1]
            })
            interpolated_pixel_results.append(lai_interpolated_df)

        # concatenate the results for all pixels
        interpolated_pixel_results_parcel = pd.concat(
            interpolated_pixel_results, ignore_index=True)
        # correct the coordinates as xarray shifts them to the center
        # we fix the pixel resolution to 10 meters (S2 resolution)
        interpolated_pixel_results_parcel['y'] = \
            interpolated_pixel_results_parcel['y'] + 5  # meters
        interpolated_pixel_results_parcel['x'] = \
            interpolated_pixel_results_parcel['x'] - 5  # meters

        sc = SceneCollection()
        for time_stamp in interpolated_pixel_results_parcel.time.unique():
            # get the data for the current time stamp
            data = interpolated_pixel_results_parcel[
                interpolated_pixel_results_parcel.time == time_stamp].copy()

            # convert to eodal RasterCollection
            # reconstruct geoinfo
            geo_info = GeoInfo(
                epsg=32632,
                ulx=data.x.min(),
                uly=data.y.max(),
                pixres_x=10,
                pixres_y=-10
            )

            data_gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data.x, data.y),
                crs=geo_info.epsg
            )
            rc = RasterCollection()
            for band_name in ['lai',
                              'lai_minus_std',
                              'lai_plus_std']:
                band = Band.from_vector(
                    vector_features=data_gdf,
                    geo_info=geo_info,
                    band_name_src=band_name,
                    band_name_dst=band_name,
                    nodata_dst=np.nan
                )
                rc.add_band(band)
            # cast date to datetime
            if covariate_granularity == 'daily':
                time_stamp = pd.to_datetime(time_stamp)
            rc.scene_properties = SceneProperties(
                acquisition_time=time_stamp
            )
            sc.add_scene(rc)

        # save the SceneCollection as pickled object
        sc = sc.sort()
        fname_pkl = output_dir.joinpath(
            f'{covariate_granularity}_lai.pkl')
        with open(fname_pkl, 'wb') as dst:
            dst.write(sc.to_pickle())

        logger.info(
            f'Interpolated {parcel_dir.name} to ' +
            f'{covariate_granularity} LAI values ' +
            f'using {response_curve_type} response curve')


if __name__ == '__main__':

    import os
    cwd = Path(__file__).absolute().parent.parent
    os.chdir(cwd)

    # apply model at the validation sites and the test sites
    # directory with parcel LAI time series
    directories = ['validation_sites']

    for directory in directories:
        parcel_lai_dir = Path('results') / directory

        dose_response_parameters = Path(
            'data/dose_reponse_in-situ/output/parameter_model')

        response_curve_types = ['WangEngels', 'asymptotic', 'non_linear']

        covariate_granularities = ['daily', 'hourly']

        # percentage of data points to be removed
        percentage_datapoints_to_remove = 0.1

        for response_curve_type in response_curve_types:
            for covariate_granularity in covariate_granularities:
                apply_temperature_response(
                    parcel_lai_dir=parcel_lai_dir,
                    dose_response_parameters=dose_response_parameters,
                    response_curve_type=response_curve_type,
                    covariate_granularity=covariate_granularity,
                    percentage_datapoints_to_remove=percentage_datapoints_to_remove,  # noqa E501
                )
