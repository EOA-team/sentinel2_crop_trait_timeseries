"""
Evaluate which pixels failed and why.

Usage:

.. code-block:: shell

    python 02_get_failed_pixels.py

@author: Lukas Valentin Graf
"""

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

plt.style.use('bmh')


def get_failed_pixels(
        directories: list[str | Path],
) -> pd.DataFrame:
    """
    Get the failed pixels (coordinates and reason)

    Parameters
    ----------
    directories : list[str | Path]
        List with directories with the failed pixels.
    Returns
    -------
    pd.DataFrame
        DataFrame with the failed pixels.
    """
    # loop over the directories
    failed_pixels_list = []
    for directory in directories:
        # loop over the parcel results
        for parcel_dir in directory.glob('parcel_*'):
            # get the failed pixels
            fpath_failed = parcel_dir / 'sigmoid' / 'failed_pixels.csv'
            if not fpath_failed.exists():
                continue
            df_failed = pd.read_csv(fpath_failed)
            df_failed['parcel'] = parcel_dir.name
            failed_pixels_list.append(df_failed)

    # concatenate the failed pixels
    df_failed_pixels = pd.concat(failed_pixels_list, ignore_index=True)
    return df_failed_pixels


def get_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Get some statistics of the failed pixels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the failed pixels.
    output_dir : Path
        Path to the output directory.
    """
    # get the errors and the number of pixels per error
    errors = df.error.value_counts()
    # plot the errors as pie chart
    fig, ax = plt.subplots()
    ax.pie(errors, labels=errors.index, autopct='%1.1f%%')
    ax.set_title(f'N = {len(df)} pixels')
    fig.savefig(output_dir / 'errors_pie-chart.png', dpi=300,
                bbox_inches='tight')


if __name__ == '__main__':

    import os
    cwd = Path(__file__).absolute().parent.parent.parent
    os.chdir(cwd)

    # apply model at the validation sites and the test sites
    # directory with parcel LAI time series
    directories = ['validation_sites', 'test_sites']
    directories = [Path('results') / x for x in directories]

    # get the failed pixels
    df_failed_pixels = get_failed_pixels(directories)

    # save the failed pixels
    output_dir = Path('analysis') / 'failed_pixels'
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath_failed_pixels = output_dir / 'failed_pixels.csv'
    df_failed_pixels.to_csv(fpath_failed_pixels, index=False)

    # get some statistics of the failed pixels
    stats = get_statistics(df_failed_pixels, output_dir=output_dir)
