"""
Utility functions for validation and data analysis.

@author: Lukas Valentin Graf
"""


import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import linregress


def calculate_error_stats(
        val_df: pd.DataFrame,
        column_x: str = 'dry_yield_normalized',
        column_y: str = 'auc'
) -> dict:
    """
    Calculate the error statistics for the given validation
    dataframe.

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation dataframe.
    column_x : str, optional
        Column name for the x-axis, by default 'dry_yield_normalized'.
    column_y : str, optional
        Column name for the y-axis, by default 'auc'.
    Returns
    -------
    dict
        Error statistics dictionary.
    """
    n = len(val_df)
    val_df.dropna(inplace=True)
    # linear regression between in-situ and model LAI
    slope, intercept, r_value, _, std_err = linregress(
        val_df[column_x], val_df[column_y])
    # calculate the bias
    prediction_variance = np.var(val_df[column_y])
    sse = np.mean((np.mean(val_df[column_y]) - val_df[column_x])**2)

    # calculate the error metrics
    error_stats = {
        'RMSE': np.sqrt(np.mean((val_df[column_x] -
                                 val_df[column_y])**2)),
        'NMAD': 1.4826 * np.median(np.abs(val_df[column_x] -
                                          val_df[column_y])),
        'MAE': np.mean(np.abs(val_df[column_x] -
                              val_df[column_y])),
        'MAPE': np.mean(np.abs((val_df[column_x] -
                                val_df[column_y]) /
                        val_df[column_x])),
        'R2': r_value**2,
        'nRMSE': np.sqrt(np.mean((val_df[column_x] -
                                  val_df[column_y])**2)) /
                    (val_df[column_x].max() -
                     val_df[column_x].min()),
        'nMAE': np.mean(np.abs(val_df[column_x] -
                               val_df[column_y])) /
                    (val_df[column_x].max() -
                     val_df[column_x].min()),
        'nMAPE': np.mean(np.abs((val_df[column_x] -
                                 val_df[column_y]) /
                         val_df[column_x])) /
                        (val_df[column_x].max() -
                         val_df[column_x].min()),
        'n': n,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err,
        'bias': sse - prediction_variance
    }
    return error_stats


def count_scenes(
        model_output_dir: Path
) -> pd.DataFrame:
    """
    Count the number of available S2 scenes.

    :param model_output_dir: directory with model outputs
    :returns: DataFrame with number of available S2 scenes
    """
    res = []
    for site_dir in model_output_dir.glob('farm_*'):
        raw_lai = pd.read_csv(site_dir.joinpath('raw_lai_values.csv'))
        n_scenes = len(raw_lai.time.unique())
        res.append({'site': site_dir.name, 'n_scenes': n_scenes})
    return pd.DataFrame(res)


"""
if __name__ == '__main__':

    model_output_dir = Path('results/validation_sites')
    scene_count_df = count_scenes(model_output_dir=model_output_dir)
    scene_count_df.to_csv(model_output_dir.joinpath('s2_scene_counts.csv'),
                          index=False)
"""
