"""
Plots maps and scatterplots of the asymptotic and
sigmoid (baseline) model for selected points in time
during stem elongation and heading to show the performance
of the models on the field scale.

This script outputs Figure 9 in the paper.

.. code-block:: shell

    python 03_glai_scatter_models_sat.py

@author: Lukas Valentin Graf
"""

import matplotlib.pyplot as plt
import mpl_scatter_density
import numpy as np
import pandas as pd

from eodal.core.scene import SceneCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

black_listed_parcels = ['Bramenwies', 'Hinteracker 2']
bbch_stages_of_interest = [(31, 33), (55, 59)]
plt.style.use('bmh')

model_mappings = {
    'non_linear': 'Non linear',
    'asymptotic': 'Asymptotic',
    'WangEngels': 'Wang Engels',
    'sigmoid': 'Baseline (sigmoid)',
    'in-situ': 'In-situ'
}
plt.style.use('bmh')

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)
# set the matplotlib fontsize to 16
plt.rcParams.update({'font.size': 16})


def main(
        model_output_dir: Path,
        validation_data_dir: Path,
        selected_models: list[str],
        output_dir: Path,
        year: int
) -> None:
    """
    Generate the scatter plots of reconstructed vs. in-situ measured GLAI.

    :param model_output_dir: directory with model outputs
    :param validation_data_dir: directory with in-situ validation data
    :param selected_models: reconstruction models to consider
    :param output_dir: output directory
    :param year: year to consider (either 2022 or 2023)
    """
    if len(selected_models) > 2:
        raise ValueError('Cannot plot more than two models at once')

    # loop over directories in the model output directory
    for site_dir in model_output_dir.glob('*'):

        # make sure the directory is a directory
        if not site_dir.is_dir():
            continue

        if not site_dir.name.startswith('farm'):
            continue

        # get the site name from the directory name
        site = site_dir.name.split('_')[1]

        # check site year against year of validation data
        site_year = int(site_dir.name.split('_')[2].split('-')[0])
        year = int(validation_data_dir.name)
        if site_year != year:
            continue

        # debug
        if site != 'SwissFutureFarm' and year != 2023:
            continue

        model_res = {}
        for model in selected_models:
            model_dir = site_dir.joinpath(model)

            # read the scene collection from pickle with the
            # daily LAI values
            fpath_scoll = model_dir.joinpath('daily_lai.pkl')
            scoll = SceneCollection.from_pickle(fpath_scoll)
            # get list of numpy arrays with daily LAI values
            lai_arrays = [
                x['lai'].values for _, x in scoll]
            # stack the arrays into a single 1d array
            if site == 'SwissFutureFarm' and year == 2023:
                lai_arrays = lai_arrays[2:]
                if model == 'asymptotic':
                    lai_arrays = [x[:, :34] for x in lai_arrays]
            lai = np.hstack(lai_arrays).flatten()
            model_res[model] = lai

        # plot the scatterplot for the site coloring the point density
        f = plt.figure(figsize=(8, 8))
        ax = f.add_subplot(111, projection='scatter_density')

        x = model_res[selected_models[1]]
        y = model_res[selected_models[0]]
        y = y[~np.isnan(x)]
        x = x[~np.isnan(x)]

        density = ax.scatter_density(
            x=x,
            y=y,
            cmap=white_viridis
        )
        # add a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        f.colorbar(
            density, cax=cax, orientation="vertical",
            label='Density [-]')

        ax.set_ylabel(
            f'GLAI {model_mappings[selected_models[0]]}' +
            r' [$m^2$ $m^{-2}$]', fontsize=16)
        ax.set_xlabel(
            f'GLAI {model_mappings[selected_models[1]]}' +
            r' [$m^2$ $m^{-2}$]', fontsize=16)
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'{site} {year}')
        # number of non nan values in both arrays
        n = np.count_nonzero(~np.isnan(model_res[selected_models[1]]))
        ax.plot([0, 7], [0, 7], color='grey', linestyle='solid',
                label='1:1 line (N=%d)' % (n, ))
        ax.legend(fontsize=16)
        f.tight_layout()
        f.savefig(
            output_dir / f'{site}_{year}.png',
            dpi=300)
        plt.close(f)


if __name__ == '__main__':

    model_output_dir = Path('results/validation_sites')
    output_dir = Path('figures')
    years = [2022, 2023]

    # go through the years and extract the data
    for year in years:
        validation_data_dir = Path('data/in-situ') / str(year)
        main(
            model_output_dir=model_output_dir,
            validation_data_dir=validation_data_dir,
            selected_models=['asymptotic', 'sigmoid'],
            output_dir=output_dir,
            year=year)
