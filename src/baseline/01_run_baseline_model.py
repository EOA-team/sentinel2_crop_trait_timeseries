"""
Baseline using a sigmoid model fitting on the raw LAI values.
The baseline model is only applied on the validation set.

@author: Lukas Valentin Graf
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import shutil
import warnings

from eodal.config import get_settings
from eodal.core.band import Band, GeoInfo
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from pathlib import Path
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')
logger = get_settings().logger
plt.style.use('bmh')


def sigmoid(
        x: np.ndarray,
        L: float = 1,
        k: float = 1,
        x0: float = 0,
        b: float = 0
) -> np.ndarray:
    """
    Sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        Input array.
     L : float
        scales the output range.
    k : float
        scales the input range.
    x0 : float
        is the sigmoid's midpoint.
    b : float
        is the bias.

    Returns
    -------
    np.ndarray
        Sigmoid of the input array.
    """
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


def fit_sigmoid(x: np.ndarray, y: np.ndarray):
    """
    Fit a sigmoid function to the data.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Dependent variable.

    Returns
    -------
    np.ndarray
        Fitted sigmoid function.
    """
    # mandatory initial guess
    p0 = [max(y), np.median(x), 1, min(y)]
    popt, _ = curve_fit(sigmoid, x, y, p0, method='lm')
    return popt


def to_doy(time: pd.Series) -> pd.Series:
    """
    Convert time stamps to day of year
    """
    int_time = time.dt.day_of_year
    # substract the start time from all time stamps
    int_time = int_time - int_time.min()
    return int_time


def plot_sigmoid(
        time_stamps: np.ndarray,
        lai_values: np.ndarray,
        lai_interpolated: np.ndarray
) -> plt.Figure:
    """
    Plot the sigmoid function fitted to the LAI values.

    Parameters
    ----------
    time_stamps : np.ndarray
        Time stamps.
    lai_values : np.ndarray
        Original satellite-derived LAI values.
    lai_interpolated : np.ndarray
        Interpolated LAI values.
    Returns
    -------
    plt.Figure
        Figure with the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    # get array with daily time stamps
    daily_time_stamps = np.arange(
        time_stamps.min(), time_stamps.max() + pd.Timedelta('1D'),
        dtype='datetime64[D]')
    # plot the original LAI values
    ax.scatter(time_stamps, lai_values, color='red', label='Satellite LAI')
    ax.plot(daily_time_stamps, lai_interpolated, label='Reconstructed LAI')
    ax.legend(loc='best')
    ax.set_xlabel('Time')
    # rotate x labels by 45 degrees
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_ylabel(r'LAI [$m^2$ $m^{-2}$]')
    return fig


def loop_pixels(
        parcel_lai_dir: Path,
        percentage_datapoints_to_remove: float = 0.1,
        n_plots: int = 20
) -> None:
    """
    Loop over pixels and apply the sigmoid model.

    Parameters
    ----------
    parcel_lai_dir : Path
        Directory with parcel LAI time series.
    percentage_datapoints_to_remove : float, optional
        Percentage of data points to remove randomly from the time series.
        Must be between 0 and 1. Default is 0.1.
    n_plots : int, optional
        Number of plots to generate (selected randomly). Default is 20.
    """
    # loop over parcels and read the data
    for parcel_dir in parcel_lai_dir.glob('*'):

        if not parcel_dir.is_dir():
            continue

        logger.info(f'Processing parcel {parcel_dir.name}')

        # make an output directory
        output_dir = parcel_dir.joinpath('sigmoid')
        output_dir.mkdir(exist_ok=True)
        fname_pkl = output_dir.joinpath('daily_lai.pkl')

        output_dir_plots = output_dir.joinpath('plots')
        output_dir_plots.mkdir(exist_ok=True)

        # for the test pixels we can use our phenology model
        fpath_relevant_phase = parcel_dir.joinpath('relevant_phase.txt')
        if fpath_relevant_phase.exists():
            with open(fpath_relevant_phase, 'r') as src:
                phase = src.read()
            if phase != 'stemelongation-endofheading':
                logger.info(
                    f'Skipping {parcel_dir.name} because it is not '
                    'in the right phenological phase')
                shutil.rmtree(output_dir)
                continue

        # leaf area index data
        fpath_lai = parcel_dir.joinpath('raw_lai_values.csv')
        lai = pd.read_csv(fpath_lai)
        lai['time'] = pd.to_datetime(
            lai['time'], utc=True).dt.floor('H')
        # convert time to doy
        lai['doys'] = to_doy(lai['time'])

        # convert DataFrame to GeoDataFrame (CRS is always UTM 32N)
        lai['x'] = lai['x'].round(0)
        lai['y'] = lai['y'].round(0)
        lai = gpd.GeoDataFrame(
            lai, geometry=gpd.points_from_xy(lai['x'], lai['y']),
            crs='EPSG:32632')
        # some geometries are shifted by 1 m in x and y direction
        # we need to correct this (still a bug in eodal 0.2.1)
        geoms = lai.geometry
        geoms_corrected_indices = []
        for unique_geom in geoms:
            # continue if the geometry is already processed
            if unique_geom in geoms_corrected_indices:
                continue
            distance = abs(unique_geom.distance(lai.geometry).sort_values())
            distance = distance[(distance > 0) & (distance < 2)]
            if distance.empty:
                continue
            # get the index of the geometry which is less than 2 m apart
            # from the unique geometry
            close_geom_indices = distance.index
            for close_geom_index in close_geom_indices:
                lai.loc[close_geom_index, 'geometry'] = unique_geom
                lai.loc[close_geom_index, 'x'] = unique_geom.x
                lai.loc[close_geom_index, 'y'] = unique_geom.y
                geoms_corrected_indices.append(close_geom_index)

        # determine randomly for which pixel_coords we want to
        # generate plots
        try:
            pixel_coords_to_plot = random.sample(
                list(lai.groupby(['y', 'x']).groups.keys()), n_plots)
        except ValueError:
            pixel_coords_to_plot = random.sample(
                list(lai.groupby(['y', 'x']).groups.keys()), 1)

        # loop over single pixels
        interpolated_pixel_results = []
        failed_coords = []
        for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):

            plot = pixel_coords in pixel_coords_to_plot

            lai_pixel_ts.sort_values(by='time', inplace=True)

            # randomly remove X percent of the data
            lai_pixel_ts.index = [x for x in range(len(lai_pixel_ts))]
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
            # remove nan values
            lai_values = np.delete(lai_values, nan_indices)

            # get doy
            doys = np.delete(
                lai_pixel_ts['doys'].values.copy(), nan_indices)
            n_days = doys.max() - doys.min()
            # normalize doy values between 0 and 1
            doys = (doys - doys.min()) / (doys.max() - doys.min())

            # fit the sigmoid model
            try:
                popt = fit_sigmoid(
                    x=doys,
                    y=lai_values
                )
            except Exception as e:
                # one error message about nans is misleading. It appears, when
                # there is just a single data point. Since this has nothing to
                # do with NaNs, we continue.
                wrong_error_msg = \
                    e.__str__() == 'array must not contain infs or NaNs' \
                    and not np.isnan(lai_values).any()
                if wrong_error_msg:
                    continue
                failed_coords.append({
                    'x': pixel_coords[1],
                    'y': pixel_coords[0],
                    'error': e})
                continue

            # apply the sigmoid model to obtain LAI values in the desired
            # granularity
            time_stamps = np.arange(0, 1, 1/n_days)
            time_stamps = np.append(time_stamps, [1])
            lai_interpolated = sigmoid(
                x=time_stamps,
                L=popt[0],
                k=popt[1],
                x0=popt[2],
                b=popt[3])

            # scale the doys back to the original dates
            first_date = lai_pixel_ts['time'].min()
            last_date = first_date + pd.Timedelta(days=n_days)
            daily_dates = pd.date_range(first_date, last_date, freq='D')

            # create a dataframe with the interpolated LAI values
            lai_interpolated_df = pd.DataFrame({
                'time': daily_dates,
                'lai': lai_interpolated,
                'y': pixel_coords[0],
                'x': pixel_coords[1]
            })
            interpolated_pixel_results.append(lai_interpolated_df)

            # optional plotting of pixel time series
            if plot:
                try:
                    orig_lai_values = \
                        lai_pixel_ts[
                            lai_pixel_ts['lai'].notnull()][['time', 'lai']]
                    f = plot_sigmoid(
                        time_stamps=orig_lai_values['time'].values,
                        lai_values=orig_lai_values['lai'].values,
                        lai_interpolated=lai_interpolated
                    )
                    f.savefig(
                        output_dir_plots.joinpath(
                            f'interpolated_lai_{pixel_coords[0]}'
                            f'_{pixel_coords[1]}_daily.png'),
                        dpi=300, bbox_inches='tight')
                    plt.close(f)
                except Exception as e:
                    logger.error(
                        'Could not plot sigmoid model for pixel ' +
                        f'{pixel_coords} ({parcel_dir.name}): {e}')
                    continue

        # concatenate the results for all pixels
        if len(interpolated_pixel_results) == 0:
            logger.error(
                f'Could not interpolate LAI for parcel {parcel_dir.name}')
            continue
        interpolated_pixel_results_parcel = pd.concat(
            interpolated_pixel_results, ignore_index=True)
        # correct the coordinates as xarray shifts them to the center
        # we fix the pixel resolution to 10 meters (S2 resolution)
        interpolated_pixel_results_parcel['y'] = \
            interpolated_pixel_results_parcel['y'] + 5  # meters
        interpolated_pixel_results_parcel['x'] = \
            interpolated_pixel_results_parcel['x'] - 5  # meters

        # save the failed coordinates to CSV file
        if len(failed_coords) > 0:
            failed_coords = pd.DataFrame(failed_coords)
            failed_coords.to_csv(
                output_dir.joinpath('failed_pixels.csv'), index=False)

        with open(output_dir.joinpath('pixel_count.txt'), 'w') as dst:
            total_pixels = len(interpolated_pixel_results) + len(failed_coords)
            dst.write(
                f'Failed pixels: {len(failed_coords)} ' +
                f'({np.round(len(failed_coords) / total_pixels * 100, 2)}%)\n')
            dst.write(
                f'Succeded pixels: {len(interpolated_pixel_results)} ' +
                f'({np.round(len(interpolated_pixel_results) / total_pixels * 100, 2)}%)\n')   # noqa: E501
            dst.write(f'Total Pixels: {total_pixels}\n')

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
            band = Band.from_vector(
                vector_features=data_gdf,
                geo_info=geo_info,
                band_name_src='lai',
                band_name_dst='lai',
                nodata_dst=np.nan
            )
            rc = RasterCollection()
            rc.add_band(band)
            rc.scene_properties = SceneProperties(
                acquisition_time=time_stamp
            )
            sc.add_scene(rc)

        # save the SceneCollection as pickled object
        sc = sc.sort()
        with open(fname_pkl, 'wb') as dst:
            dst.write(sc.to_pickle())

        logger.info(f'Interpolated {parcel_dir.name} to daily LAI values')


if __name__ == '__main__':

    import os
    cwd = Path(__file__).absolute().parent.parent.parent
    os.chdir(cwd)

    # apply model at the validation sites and the test sites
    # directory with parcel LAI time series
    directories = ['validation_sites']  # 'validation_sites', 'test_sites'
    percentage_datapoints_to_remove = 0.1

    for directory in directories:
        parcel_lai_dir = Path('results') / directory
        loop_pixels(
            parcel_lai_dir=parcel_lai_dir,
            percentage_datapoints_to_remove=percentage_datapoints_to_remove
        )
