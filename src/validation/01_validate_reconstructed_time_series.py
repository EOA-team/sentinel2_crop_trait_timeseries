"""
Validate the reconstructed LAI time series. This scripts
produces the subplots shown in Figure 6 in the paper.

.. code-block:: shell

    python 01_validate_reconstructed_time_series.py

@author: Lukas Valentin Graf
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from copy import deepcopy
from eodal.core.scene import SceneCollection
from pathlib import Path

from utils import calculate_error_stats


models = ['non_linear', 'sigmoid', 'asymptotic', 'WangEngels']
bbch_range = (30, 59)
black_listed_parcels = ['Bramenwies']

plt.style.use('bmh')
warnings.filterwarnings('ignore')

model_mappings = {
    'non_linear': 'Non linear',
    'asymptotic': 'Asymptotic',
    'WangEngels': 'Wang Engels',
    'sigmoid': 'Baseline (sigmoid)'
}


def get_data(
        model_output_dir: Path,
        validation_data_dir: Path,
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

        # get the site name from the directory name
        site = site_dir.name.split('_')[1]

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

                scoll = SceneCollection.from_pickle(fpath_scoll)
                min_date = pd.to_datetime(scoll.timestamps[0], utc=True)
                max_date = pd.to_datetime(scoll.timestamps[-1], utc=True)
                # loop over dates for which in-situ data is available
                # and extract the interpolated LAI values
                pixel_vals_list = []
                for date, site_val_df_date in site_val_df.groupby('date'):
                    if granularity == 'hourly':
                        date_rounded = date.round('H').tz_convert('UTC')
                    elif granularity == 'daily':
                        date_rounded = pd.Timestamp(date.date()).tz_localize(
                            'Europe/Zurich').tz_convert('UTC')
                    # continue if date is not between min and max date
                    if not (min_date <= date_rounded <= max_date):
                        continue

                    # get the interpolated LAI value by timestamp
                    if granularity == 'hourly':
                        scene = scoll[date_rounded.__str__()]
                    elif granularity == 'daily':
                        date_date = date_rounded.date()
                        # get the corresponding date in the scene collection
                        for timestamp in scoll.timestamps:
                            if pd.to_datetime(timestamp).date() == date_date:
                                scene = scoll[timestamp]
                                break

                    # get the pixel values at the in-situ points
                    pixel_vals = scene.get_pixels(
                        vector_features=site_val_df_date)
                    pixel_vals = pixel_vals.rename(
                        columns={'lai': f'lai_{model}'})
                    pixel_vals_list.append(pixel_vals)

                # concatenate the pixel values
                if len(pixel_vals_list) == 0:
                    continue
                pixel_vals_df = pd.concat(pixel_vals_list)
                pixel_vals_df['model'] = model
                pixel_vals_df['granularity'] = granularity

                # save the pixel values
                fpath_out = model_dir / f'{granularity}_lai_validation.csv'
                pixel_vals_df.to_csv(fpath_out, index=False)

                print(f'{site} {model} {granularity} --> done')


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
    with sns.plotting_context("notebook", font_scale=1.2):
        sns.scatterplot(
            data=val_df,
            x='lai_in-situ',
            y=f'lai_{error_stats["model"]}',
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
    ax.set_ylabel(r'model GLAI [$m^2$ $m^{-2}$]', fontsize=18)
    # add a text box with the error statistics
    textstr = '\n'.join((
        r'$N=%d$' % (error_stats['n'], ),
        r'$RMSE=%.2f$ $m^2$ $m^{-2}$' % (error_stats['RMSE'], ),
        r'$nRMSE=%.2f$' % (error_stats['nRMSE'] * 100, ) + '%',
        r'$Bias=%.2f$ $m^2$ $m^{-2}$' % (error_stats['bias'], ),
        r'$R^2=%.2f$' % (error_stats['R2'], )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.6)
    ax.text(3.45, 0.2, textstr, fontsize=13, bbox=props)
    # set the axis limits
    ax.set_xlim(min_lai, max_lai)
    ax.set_ylim(min_lai, max_lai)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    # set the title
    model = list(val_df.model.unique())[0]
    ax.set_title(f'{model_mappings[model]} - {error_stats["granularity"]}',
                 fontsize=18)


def combine_validation_files(
      model_output_dir: Path,
      model_results: dict,
      years_to_ignore: list = [],
      bbch_range: tuple = (30, 59)
) -> dict:
    """
    Combine the validation files for all sites and years.

    :param model_output_dir: directory with model outputs.
    :param model_results: empty dictionary for storing validation data
    :param years_to_ignore: optional list of years to ignore for validation
    :param bbch_range: BBCH range to consider
    :returns: populated model results 
    """
    for site_dir in model_output_dir.glob('farm_*'):
        site = site_dir.name.split('_')[1]
        year = int(site_dir.name.split('_')[2].split('-')[0])
        if year in years_to_ignore:
            continue
        for model in models:
            for granularity in granularities:
                fpath_validation = site_dir / model / \
                    f'{granularity}_lai_validation.csv'
                if not fpath_validation.exists():
                    continue
                # read the validation data
                val_df = pd.read_csv(fpath_validation)
                # filter by BBCH range
                val_df = val_df[
                    (val_df['BBCH Rating'] >= bbch_range[0]) &
                    (val_df['BBCH Rating'] <= bbch_range[1])
                ].copy()
                val_df['site'] = site
                val_df['year'] = int(year)
                model_results[model][granularity].append(val_df)
    return model_results


def validate(
        model_output_dir: Path,
        model_results: dict,
        models: list = ['sigmoid', 'non_linear', 'asymptotic'],
        granularities: list = ['hourly', 'daily'],
        years: list = [2022, 2023],
        bbch_range: tuple = (30, 59)
) -> pd.DataFrame:
    """
    Validate the LAI models.

    :param model_output_dir: directory with model outputs.
    :param model_results: dictionary with validation data
    :param models: list of models to consider
    :param granularities: list of temporal granularities (hourly, daily) to consider
    :param year: list of years to consider (2022, 2023)
    :param bbch_range: BBCH range to consider (30 to 59)
    :returns: DataFrame with validation results (error statistics)
    """
    validation_results = []
    for model in models:
        f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
        for idx, granularity in enumerate(granularities):
            # concatenate the dataframes
            if len(model_results[model][granularity]) == 0:
                # the sigmoid model does not have hourly data
                ax[idx].remove()
                continue
            val_df = pd.concat(model_results[model][granularity])

            # calculate the error statistics
            error_stats = calculate_error_stats(
                val_df,
                column_x='lai_in-situ',
                column_y=f'lai_{model}')
            error_stats['model'] = model
            error_stats['granularity'] = granularity
            validation_results.append(error_stats)
            # plot the results
            plot_results(val_df, error_stats, ax[idx])

        # save the figure
        year_str = '-'.join([str(year) for year in years])
        fname_plot = model_output_dir.joinpath(
            f'{model}_validation_{year_str}_' +
            f'{bbch_range[0]}-{bbch_range[1]}.png')
        f.savefig(fname_plot, bbox_inches='tight', dpi=300)
        plt.close(f)

    # save the validation results to a CSV file
    validation_results_df = pd.DataFrame(validation_results)
    return validation_results_df


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
        get_data(
            model_output_dir=model_output_dir,
            validation_data_dir=validation_data_dir)

    # carry out the actual validation part
    models = ['non_linear', 'sigmoid', 'asymptotic', 'WangEngels']
    granularities = ['hourly', 'daily']
    # create dictionary to store the results
    model_results = dict.fromkeys(models)
    for model in models:
        model_results[model] = {}
        for granularity in granularities:
            model_results[model].update({granularity: []})
    model_results_save = deepcopy(model_results)

    # loop over directories in the model output directory
    # to get the validation data
    model_results = combine_validation_files(
        model_output_dir=model_output_dir,
        model_results=model_results)

    # loop over the models and granularities and calculate the
    # error statistics and plot the results
    validation_results_df = validate(
        model_output_dir=model_output_dir,
        model_results=model_results,
        models=models)
    validation_results_df.to_csv(
        model_output_dir / 'validation_results_2022-2023.csv',
        index=False)

    # validate by BBCH macro-stages (30-39, 40-49, 50-59)
    bbch_ranges = [(30, 39), (40, 49), (50, 59)]
    for bbch_range in bbch_ranges:
        model_results = deepcopy(model_results_save)
        model_results = combine_validation_files(
            model_output_dir=model_output_dir,
            model_results=model_results,
            bbch_range=bbch_range)
        validation_results_df = validate(
            model_output_dir=model_output_dir,
            model_results=model_results,
            models=models,
            bbch_range=bbch_range)
        validation_results_df.to_csv(
            model_output_dir.joinpath(
                f'validation_results_{bbch_range[0]}-{bbch_range[1]}.csv'),
            index=False)

    # validate by year
    years_to_ignore = [2022, 2023]
    for year_to_ignore in years_to_ignore:
        year_included = [year for year in years if year != year_to_ignore]
        model_results = deepcopy(model_results_save)
        model_results = combine_validation_files(
            model_output_dir=model_output_dir,
            model_results=model_results,
            years_to_ignore=[year_to_ignore])
        validation_results_df = validate(
            model_output_dir=model_output_dir,
            model_results=model_results,
            years=year_included,
            models=models)
        validation_results_df.to_csv(
            model_output_dir.joinpath(f'validation_results_{year_included[0]}.csv'),
            index=False)
