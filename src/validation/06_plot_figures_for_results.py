"""
Plot figures for the Results section of the paper.

This script outputs Figures 6, 7, and 8 in the paper.

.. code-block:: shell

    python 06_plot_figures_for_results.py

@author: Lukas Valentin Graf
"""

from pathlib import Path

import matplotlib.pyplot as plt


def scatter_plots(result_dir: Path) -> plt.Figure:

    fnames_png = [
        'non_linear_validation_2022-2023_30-59.png',
        'asymptotic_validation_2022-2023_30-59.png',
        'WangEngels_validation_2022-2023_30-59.png',
        'sigmoid_validation_2022-2023_30-59.png'
    ]
    fpaths_pngs = [result_dir.joinpath(x) for x in fnames_png]
    imgs = [plt.imread(x) for x in fpaths_pngs]

    f, ax = plt.subplots(nrows=2, ncols=2,
                         figsize=(11, 6),
                         sharex=False, sharey=False)
    ax = ax.flatten()
    titles = ['(a)', '(b)', '(c)', '(d)']
    for idx in range(len(fpaths_pngs)):
        ax[idx].imshow(imgs[idx])
        ax[idx].axis('off')
        ax[idx].set_title(titles[idx], fontsize=18, loc='left')

    return f


def errors_with_bbch(result_dir: Path) -> plt.Figure:

    fnames_png = [
        'RMSE_daily_bbch-stages.png',
        'nRMSE_daily_bbch-stages.png',
        'bias_daily_bbch-stages.png',
        'R2_daily_bbch-stages.png'
    ]
    fpaths_pngs = [result_dir.joinpath(x) for x in fnames_png]
    imgs = [plt.imread(x) for x in fpaths_pngs]

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    ax = ax.flatten()
    titles = ['(a) RMSE', '(b) nRMSE', '(c) Bias', r'(d) $R^2$']
    for idx in range(len(fpaths_pngs)):
        ax[idx].imshow(imgs[idx])
        ax[idx].axis('off')
        ax[idx].set_title(titles[idx], fontsize=18, loc='left')

    return f


def temporal_profiles(result_dir: Path) -> plt.Figure:

    fnames_png = [
        'farm_Strickhof_2022-03-20-2022-06-11/Fluegenrain_temporal_profile.png',
        'farm_Strickhof_2022-03-20-2022-06-11/Hohrueti_temporal_profile.png',
        'farm_SwissFutureFarm_2022-04-06-2022-06-14/Altkloster_temporal_profile.png',
        'farm_SwissFutureFarm_2022-04-06-2022-06-14/Ruetteli_temporal_profile.png',
        'farm_Strickhof_2023-02-23-2023-06-04/Neuhof 3_temporal_profile.png',
        'farm_Strickhof_2023-02-23-2023-06-04/Hohrueti 2_temporal_profile.png',
        'farm_SwissFutureFarm_2023-03-05-2023-06-04/Grund_temporal_profile.png'
    ]
    fpaths_pngs = [result_dir.joinpath(x) for x in fnames_png]
    imgs = [plt.imread(x) for x in fpaths_pngs]

    f, ax = plt.subplots(nrows=4, ncols=2, figsize=(18, 20))
    ax = ax.flatten()
    titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
    for idx in range(len(fpaths_pngs)):
        ax[idx].imshow(imgs[idx])
        ax[idx].axis('off')
        ax[idx].set_title(titles[idx], fontsize=18, loc='left')
    ax[-1].axis('off')
    return f


if __name__ == '__main__':

    result_dir = Path('results/validation_sites')
    output_dir = Path('.').joinpath('figures')
    output_dir.mkdir(exist_ok=True)

    # plot the scatterplot of GLAI values (modeled vs. in-situ)
    f_scatter = scatter_plots(result_dir=result_dir)
    f_scatter.tight_layout()
    f_scatter.savefig(output_dir.joinpath('glai_scatter_plots.png'),
                      dpi=300,
                      bbox_inches='tight')
    plt.close(f_scatter)

    # plot the error measures and their change with BBCH macro-stages
    f_errors = errors_with_bbch(
        result_dir=result_dir.joinpath('error_stats_plots'))
    f_errors.tight_layout()
    f_errors.savefig(output_dir.joinpath('glai_daily-error_plots-bbch.png'),
                     dpi=300,
                     bbox_inches='tight')
    plt.close(f_errors)

    # temporal profiles
    f_profiles = temporal_profiles(result_dir=result_dir)
    f_profiles.tight_layout()
    f_profiles.savefig(output_dir.joinpath('glai_daily-temporal_profiles.png'),
                       dpi=300,
                       bbox_inches='tight')
    plt.close(f_profiles)
