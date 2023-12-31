'''
Trait extraction from Sentinel-2 imagery by radiative transfer model inversion.
The required lookup tables have been generated in the previous step (i.e.,
`01_extract_s2_data.py`).

Usage:

.. code-block:: shell

    python 02_extract_s2_traits.py

@author: Lukas Valentin Graf
'''

import numpy as np
import pandas as pd
import shutil
import warnings

from eodal.config import get_settings
from eodal.core.band import Band
from eodal.core.raster import RasterCollection
from pathlib import Path
from typing import Dict, List

from rtm_inv.core.inversion import inv_img, retrieve_traits


logger = get_settings().logger
warnings.filterwarnings('ignore')

band_selection = [
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']


def extract_s2_traits(
    data_dir: Path,
    farms: List[str],
    n_solutions: Dict[str, int],
    cost_functions: Dict[str, str],
    aggregation_methods: Dict[str, str],
    lut_sizes: Dict[str, str]
) -> None:
    """
    Lookup table based inversion of Sentinel-2 imagery.

    :param data_dir:
        directory with Sentinel-2 imagery
    :param farms:
        list of farm names
    :param n_solutions:
        number of solutions to retrieve
    :param cost_functions:
        cost functions to use for inversion
    :param aggregation_methods:
        aggregation methods to use for inversion
    :param lut_sizes:
        number of LUT entries to use for inversion
    """
    # loop over locations
    for farm in farms:
        farm_dir = data_dir.joinpath(farm)
        if not farm_dir.exists():
            continue
        # loop over scenes in farm, find lookup tables and apply the LUT based
        # inversion
        for scene_dir in farm_dir.glob('*.SAFE'):
            # load the Sentinel-2 data
            fpath_s2_raster = scene_dir.joinpath('SRF_S2.tiff')

            s2_ds = RasterCollection.from_multi_band_raster(
                fpath_raster=fpath_s2_raster)
            bands = s2_ds.band_names
            bands.remove('SCL')

            # The border regions are not recognized correctly
            # resulting in a wrong LAI
            mask_values = [0, 1.e+20]
            for mask_value in mask_values:
                mask_blue = s2_ds['B02'] == mask_value
                s2_ds.mask(mask=mask_blue, inplace=True)
            s2_spectra = s2_ds.get_values(band_selection=band_selection)

            logger.info(f'{farm}: Started inversion of {scene_dir.name}')
            # find the LUTs generated and use them for inversion
            for fpath_lut in scene_dir.glob('*lut.pkl'):
                lut = pd.read_pickle(fpath_lut)
                pheno_phase = fpath_lut.name.split('_')[0]
                if pheno_phase == 'all':
                    pheno_phase = 'all-phases'

                # check if inversion results exists already
                fname = scene_dir.joinpath(f'{pheno_phase}_lutinv_traits.tiff')
                if fname.exists():
                    logger.info(f'{farm}: {fname.name} already exists')
                    continue

                # draw sub-sample from LUT if required
                if lut_sizes[pheno_phase] < lut.shape[0]:
                    lut = lut.sample(lut_sizes[pheno_phase])

                # invert the S2 scene by comparing ProSAIL simulated to S2
                # observed spectra
                s2_lut_spectra = lut[bands].values

                if isinstance(s2_spectra, np.ma.MaskedArray):
                    mask = s2_spectra.mask[0, :, :]
                    s2_spectra = s2_spectra.data
                else:
                    mask = np.zeros(
                        shape=(s2_spectra.shape[1], s2_spectra.shape[2]),
                        dtype='uint8')
                    mask = mask.astype('bool')
                    mask[s2_spectra[0, :, :] == 0] = True

                lut_idxs, cost_function_values = inv_img(
                    lut=s2_lut_spectra,
                    img=s2_spectra,
                    mask=mask,
                    cost_function=cost_functions[pheno_phase],
                    n_solutions=n_solutions[pheno_phase],
                )
                trait_img, q05_img, q95_img = retrieve_traits(
                    lut=lut,
                    lut_idxs=lut_idxs,
                    traits=['lai', 'ccc'],
                    cost_function_values=cost_function_values,
                    measure=aggregation_methods[pheno_phase]
                )

                # save traits to file
                trait_collection = RasterCollection()
                for tdx, trait in enumerate(['lai', 'ccc']):
                    trait_collection.add_band(
                        Band,
                        geo_info=s2_ds['B02'].geo_info,
                        band_name=trait,
                        values=trait_img[tdx, :, :]
                    )
                    trait_collection.add_band(
                        Band,
                        geo_info=s2_ds['B02'].geo_info,
                        band_name=f'{trait}_q05',
                        values=q05_img[tdx, :, :]
                    )
                    trait_collection.add_band(
                        Band,
                        geo_info=s2_ds['B02'].geo_info,
                        band_name=f'{trait}_q95',
                        values=q95_img[tdx, :, :]
                    )
                # save lowest, median and highest cost function value
                highest_cost_function_vals = cost_function_values[-1, :, :]
                highest_cost_function_vals[np.isnan(trait_img[0, :, :])] = \
                    np.nan
                lowest_cost_function_vals = cost_function_values[0, :, :]
                lowest_cost_function_vals[np.isnan(trait_img[0, :, :])] = \
                    np.nan
                median_cost_function_vals = np.median(
                    cost_function_values[:, :, :], axis=0)
                median_cost_function_vals[np.isnan(trait_img[0, :, :])] = \
                    np.nan
                trait_collection.add_band(
                    Band,
                    geo_info=s2_ds['B02'].geo_info,
                    band_name='lowest_error',
                    values=lowest_cost_function_vals
                )
                trait_collection.add_band(
                    Band,
                    geo_info=s2_ds['B02'].geo_info,
                    band_name='highest_error',
                    values=highest_cost_function_vals
                )
                trait_collection.add_band(
                    Band,
                    geo_info=s2_ds['B02'].geo_info,
                    band_name='median_error',
                    values=median_cost_function_vals
                )
                # save to GeoTiff
                trait_collection.to_rasterio(fpath_raster=fname)

            logger.info(f'{farm}: Finished inversion of {scene_dir.name}')


if __name__ == '__main__':

    # list of farms to process
    farms = ['Strickhof', 'SwissFutureFarm']
    data_dir = Path('data/S2_Traits')

    # inversion set-up for the different phenological phases
    cost_functions = {
        'all-phases': 'mae',
        'germination-endoftillering': 'rmse',
        'stemelongation-endofheading': 'mae',
        'flowering-fruitdevelopment-plantdead': 'mae'
    }
    aggregation_methods = {
        'all-phases': 'median',
        'germination-endoftillering': 'median',
        'stemelongation-endofheading': 'median',
        'flowering-fruitdevelopment-plantdead': 'median'
    }
    n_solutions = {
        'all-phases': 5000,
        'germination-endoftillering': 100,
        'stemelongation-endofheading': 5000,
        'flowering-fruitdevelopment-plantdead': 5000
    }
    lut_sizes = {
        'all-phases': 50000,
        'germination-endoftillering': 10000,
        'stemelongation-endofheading': 50000,
        'flowering-fruitdevelopment-plantdead': 50000
    }

    extract_s2_traits(
        data_dir,
        farms,
        n_solutions=n_solutions,
        cost_functions=cost_functions,
        aggregation_methods=aggregation_methods,
        lut_sizes=lut_sizes
    )
