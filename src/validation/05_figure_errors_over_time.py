"""
Plot errors over time (e.g., by phenological macro-stage)

Usage:

.. code-block:: shell

    python 05_figures_errors_over_time.py

@author: Lukas Valentin Graf
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path

plt.style.use('bmh')

bbch_stages = ['30-39', '50-59']  # booting was omitted
metrics = ['RMSE', 'MAE', 'R2', 'nRMSE', 'bias']
metric_units = {
    'RMSE': r'$m^2$ $m^{-2}$',
    'MAE': r'$m^2$ $m^{-2}$',
    'R2': r'-',
    'nRMSE': r'%',
    'bias': r'$m^2$ $m^{-2}$'
}
model_mappings = {
    'non_linear': 'Non linear',
    'asymptotic': 'Asymptotic',
    'WangEngels': 'Wang Engels',
    'sigmoid': 'Baseline (sigmoid)'
}


def plot_errors_over_time(result_dir: Path):

    output_dir = result_dir.joinpath('error_stats_plots')
    output_dir.mkdir(exist_ok=True)

    # loop over CSV files with error statistics
    error_stats_list = []
    for bbch_stage in bbch_stages:
        fpath_error_stats = result_dir.joinpath(
            f'validation_results_{bbch_stage}.csv'
        )
        df = pd.read_csv(fpath_error_stats)
        df['BBCH Stage'] = bbch_stage
        error_stats_list.append(df)
    # concatenate error statistics
    errors = pd.concat(error_stats_list)

    # plot the data
    errors_daily = errors[errors.granularity == 'daily'].copy()
    errors_daily['model'] = errors_daily['model'].map(model_mappings)
    for metric in metrics:
        f, ax = plt.subplots(figsize=(5, 5))
        if metric == 'nRMSE':
            errors_daily[metric] *= 100
        # make very small bars connecting the dots with x axis
        sns.barplot(
            x='BBCH Stage', y=metric, hue='model', data=errors_daily,
            ax=ax, hue_order=model_mappings.values(), dodge=True,
            alpha=1, linewidth=0.4, order=bbch_stages, errwidth=0,
            width=0.5, palette='colorblind')

        if metric == 'R2':
            ax.legend(fontsize=14)
        else:
            ax.get_legend().remove()
        ax.set_xlabel('BBCH Stages', fontsize=18)
        ax.set_ylabel(f'{metric} [{metric_units[metric]}]', fontsize=18)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        f.savefig(
            output_dir.joinpath(f'{metric}_daily_bbch-stages.png'),
            bbox_inches='tight')
        plt.close(f)


if __name__ == '__main__':

    result_dir = Path('results/validation_sites')
    plot_errors_over_time(result_dir=result_dir)
