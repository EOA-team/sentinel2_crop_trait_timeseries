"""
Validate the S2 GLAI observations against the in-situ data.

This script outputs Figure 5 in the paper.

.. code-block:: shell

    python 00_validate_s2_glai_observations.py

@author: Lukas Valentin Graf
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import calculate_error_stats


target_crs = 'EPSG:32632'  # same as S2 data
bbch_range = (30, 59)
plt.style.use('bmh')


def get_insitu_data(
        insitu_data_dir: Path,
        years: list[int]
) -> gpd.GeoDataFrame:
    """
    Get the in-situ data for the validation sites

    :param insitu_data_dir:
        directory containing the in-situ data
    :param years:
        years of interest
    :return:
        GeoDataFrame containing the in-situ data
    """
    gdf_list = []
    for year in years:
        year_dir = insitu_data_dir / str(year)
        glai_file = year_dir / 'in-situ_glai.gpkg'
        glai = gpd.read_file(glai_file, parse_dates=True)
        bbch_file = year_dir / 'in-situ_bbch.gpkg'
        bbch = gpd.read_file(bbch_file, parse_dates=True)
        # join the two dataframes
        insitu = gpd.sjoin(
            left_df=glai,
            right_df=bbch[['geometry', 'BBCH Rating']],
            how='inner'
        )
        insitu.drop(columns=['index_right'], inplace=True)
        insitu.dropna(subset=['lai'], inplace=True)
        # convert to target crs
        insitu.to_crs(target_crs, inplace=True)
        # convert timestamps to dates
        insitu['date'] = pd.to_datetime(insitu['date']).dt.date

        gdf_list.append(insitu)
    # return the concatenated dataframe
    return pd.concat(gdf_list, ignore_index=True)


def plot_results(
        val_df: pd.DataFrame,
        error_stats: dict,
        ax: plt.axes,
        min_lai: float = 0,
        max_lai: float = 7
) -> None:
    """
    Plot the validation results (scatter plot and regression line).

    :param val_df: DataFrame with model and in-situ measured data
    :param error_stats: dictionary with error statistics
    :param ax: axes object to use for plotting
    :param min_lai: minimum GLAI to set axes limits. Default is 0.
    :param max_lai: maximum GLAI to set axes limits. Default is 7.
    """
    val_df['year'] = val_df['date_sat'].apply(lambda x: x.year)
    with sns.plotting_context("notebook", font_scale=1.2):
        sns.scatterplot(
            data=val_df,
            x='lai_insitu',
            y='lai_sat',
            style='year',
            hue='year',
            palette='colorblind',
            ax=ax)
    # plot the regression line
    x = np.linspace(min_lai,
                    max_lai, 100)
    y = error_stats['slope'] * x + error_stats['intercept']
    ax.plot(x, y, color='black', linestyle='--', label='regression line')
    # plot the 1:1 line
    ax.plot(x, x, color='grey', linestyle='-', label='1:1 line')
    # set the axis labels
    ax.set_xlabel(r'in-situ GLAI [$m^2$ $m^{-2}$]', fontsize=18)
    ax.set_ylabel(r'S2 GLAI [$m^2$ $m^{-2}$]', fontsize=18)
    # add a text box with the error statistics
    textstr = '\n'.join((
        r'$N=%d$' % (error_stats['n'], ),
        r'$RMSE=%.2f$ $m^2$ $m^{-2}$' % (error_stats['RMSE'], ),
        r'$nRMSE=%.2f$' % (error_stats['nRMSE'] * 100, ) + '%',
        r'$Bias=%.2f$ $m^2$ $m^{-2}$' % (error_stats['bias'], ),
        r'$R^2=%.2f$' % (error_stats['R2'], )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.6)
    ax.text(3.45, 0.2, textstr, fontsize=16, bbox=props)
    # set the axis limits
    ax.set_xlim(min_lai, max_lai)
    ax.set_ylim(min_lai, max_lai)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)


def validate(
        model_output_dir: Path,
        insitu: gpd.GeoDataFrame,
        output_dir: Path
) -> None:
    """
    Carry out the validation.

    :param model_output_dir: directory with model outputs
    :param insitu: GeoDataFrame with in-situ data
    :param output_dir: output directory
    """
    # loop over model output directories
    res_list = []
    for site_dir in model_output_dir.glob('farm_*'):
        # read the raw lai values csv file
        sat_lai_file = site_dir / 'raw_lai_values.csv'
        sat_lai = pd.read_csv(sat_lai_file)
        sat_lai['date'] = pd.to_datetime(
            sat_lai.time, format='ISO8601', utc=True).dt.date

        # convert to GeoDataFrame so we can use the spatial
        # join functionality
        sat_lai_gdf = gpd.GeoDataFrame(
            sat_lai,
            geometry=gpd.points_from_xy(
                sat_lai['x'],
                sat_lai['y']
            ),
            crs=target_crs
        )
        # we loop over the insitu data dates and see if we have any
        # satellite observations for that date
        for date in insitu['date'].unique():
            # select the in-situ data for the current date with a
            # tolerance of +/- 1 day
            date_minus_1 = date - pd.Timedelta(days=1)
            date_plus_1 = date + pd.Timedelta(days=1)
            insitu_date = insitu[insitu.date == date].copy()

            # check, if there is any satellite data for the current date
            sat_lai_date = sat_lai_gdf[
                (sat_lai_gdf['date'] >= date_minus_1) &
                (sat_lai_gdf['date'] <= date_plus_1)
            ].copy()

            if sat_lai_date.empty:
                continue
            if sat_lai_date.date.unique().size > 1:
                # if there is more than one observation, we select the
                # the first one
                sat_lai_date = sat_lai_date[
                    sat_lai_date.date == sat_lai_date.date.min()].copy()

            # next, we select those in-situ observations that are
            # spatially close to the satellite pixels
            # (within <5 m, half a pixel size)
            joined = gpd.sjoin_nearest(
                left_df=sat_lai_date,
                right_df=insitu_date,
                lsuffix='sat',
                rsuffix='insitu',
                max_distance=4.9
            )
            if joined.empty:
                continue
            joined = joined[[
                'date_sat', 'x', 'y', 'lai_sat', 'lai_insitu', 'BBCH Rating',
                'location']].copy()
            res_list.append(joined)

    # concatenate the results
    res = pd.concat(res_list, ignore_index=True)
    # save the results
    res.to_csv(model_output_dir / 's2_glai-obs_validation.csv', index=False)
    # calculate the error statistics
    error_stats = calculate_error_stats(
        res, column_x='lai_insitu', column_y='lai_sat')
    # save error statistics as csv, conver the dict to a dataframe first
    pd.DataFrame.from_dict(error_stats, orient='index').to_csv(
        model_output_dir / 's2_glai-obs_error_stats.csv')
    # plot the scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_results(res, error_stats, ax)
    fig.savefig(output_dir / 's2_glai-obs_validation.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':

    model_output_dir = Path('results/validation_sites')
    insitu_data_dir = Path('data/in-situ')

    output_dir = Path('figures')
    years = [2022, 2023]

    # get the in-situ data first
    insitu = get_insitu_data(insitu_data_dir, years)
    # clip to the BBCH range
    insitu = insitu[
        (insitu['BBCH Rating'] >= bbch_range[0]) &
        (insitu['BBCH Rating'] <= bbch_range[1])
    ].copy()

    validate(model_output_dir, insitu, output_dir)
