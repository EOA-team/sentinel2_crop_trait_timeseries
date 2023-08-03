# Sentinel2 Crop Trait Time Series Reconstruction

This paper contains code and data required to reproduce the results 
of our paper `"Probabilistic assimilation of optical satellite data with physiologically based growth functions improves crop trait time series reconstruction"`.

## Citation

```latex
@article{graf_reconstruction_2023
    title = {Probabilistic assimilation of optical satellite data with physiologically based growth functions improves crop trait time series reconstruction},
    year = {2023},
    author = {Graf, Lukas Valentin and Tschurr, Flavian and Aasen, Helge and Walter, Achim},
    journal = {under review}
}
```

## Content
The Python and R source code can be found in [src](src).

[scripts_dose_response](src/scripts_dose_response/) contains the R scripts required to fit the dose-response curves based on in-situ Green Leaf Area Index data. The fitted function parameters can be found [here](data/dose_reponse_in-situ/output/).

The main Python scripts to reconstruct the Green Leaf Area Index time series from Sentinel-2 observations at the validation sites are:

- [01_extract_s2_data.py](src/01_extract_s2_data.py) to fetch the Sentinel-2 data from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) and generate the PROSAIL lookup-tables.
- [02_extract_s2_traits.py](src/02_extract_s2_traits.py) to retrieve the traits by lookup-table inversion
- [03_generate_raw_s2_trait_trajectories.py](src/03_generate_raw_s2_trait_trajectories.py) to extract the "raw" Sentinel-2 Green Leaf Area Index trajectories.
- [04_reconstruct_s2_traits.py](src/04_reconstruct_s2_traits.py) to reconstruct the time series using dose-response response functions and Ensemble Kalman filtering.
- [validation](src/validation/) to validate the reconstructed time series against in-situ observations.

The results of the scripts will be written to [results](results/)


## Data

The [in-situ data](data/in-situ/) used in this work is the result of hard work by a lot of people in the field and laboratory conducted by teams at [ETH Zurich Crop Science](https://kp.ethz.ch/), the [School of Agircultural, Forest and Food Sciences, HAFL](https://www.bfh.ch/hafl/en/) and the [Division of Agroecology and Environment at Agroscope Reckenholz](https://www.agroscope.admin.ch/agroscope/en/home/about-us/organization/competence-divisions-strategic-research-divisions/agroecology-environment.html). A [list of contributors](data/AUTHORS.txt) is provided.

We therefore kindly ask you to **acknowledge our work** by

* **citing** our research properly whenever you use the data and/or methods presented here
* leave a **star on GitHub** and/or fork our repository

This helps us to continue the labor and cost-intensive process of data acquisition, preparation and, ultimately, publication to benefit science and society.

If your work relies substantially on our data please also [get in touch with us](https://www.eoa-team.net/) and consider offering co-authorship.

## Known Issues

You must use `Pandas 2.x`, older versions of `Pandas` (`Pandas 1.x`) will cause errors.
