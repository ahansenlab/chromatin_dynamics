# Code for "Chromatin Dynamics are Highly Subdiffusive Across Seven Orders of Magnitude"

This repository contains the code used in our chromatin dynamics paper (_in
prep_). The associated data is available on
[Zenodo](https://doi.org/10.5281/zenodo.15369544).

Getting started
---------------
1) __Clone this repo__ and move into it
```
$ git clone https://github.com/ahansenlab/chromatin_dynamics
$ cd chromatin_dynamics
```

2) Most of the code in this repository requires the __data__ used in the study,
which can be found at [ZENODO]. Download what you need (see below) and extract
the `.zip` files.  Store the data in subdirectories corresponding to the names
of the `.zip` files, breaking at double underscores. So, for example, the
contents of `raw_data__MINFLUX__production__npy.zip` should go to
`raw_data/MINFLUX/production/npy`; the contents of `data.zip` go into `data/`;
etc. This is where the code expects to find its data (which can of course be
changed).
```
# unzip archives from Zenodo, breaking into subdirectories at double underscores
$ mkdir -p raw_data/MINFLUX/production
$ unzip raw_data__MINFLUX__production__npy.zip -d raw_data/MINFLUX/production/npy
$ unzip data.zip
```

3) Set up a python __virtual environment and run Jupyter Notebook__. Use python
3.9 for best results (as of writing, `bayesmsd` has its cython extension
compiled only for python 3.9; using other versions will make fitting unbearably
slow. If you're not interested in the Bayesian MSD fitting, use your favorite
python version).
```
$ python3 -m venv new-environment
$ source new-environment/bin/activate
$ pip install -r requirements.txt
$ jupyter notebook
```

4) Execute the notebooks you're curious about and enjoy exploring / development!

What's happening where?
-----------------------
- `01_L-sweep` : processing and analysis of our experiments to identify a suitable probing pattern diameter L. Relies on raw data from Zenodo.
- `02_data_processing` : processing for production data set. Relies on raw data from Zenodo.
- `03_fitting` : running Bayesian MSD fits
- `04_Fig3_plots` : recreate panels for Fig. 3 of the paper
- `05_Fig4_plots` : recreate panels for Fig. 4 of the paper
- `06_FPT_simulations` : sampling fBm to simulate first passage times. This is an independent set of code and has its own environment specification and README.
- `07_FPT_simulation_analysis` : recreate panel 4E and associated supplementary plots.

Except for `01` and `02`, the code does not require anything labelled `raw_data` from Zenodo. The `data.zip` deposited there contains an assembled version of the complete data set, which the rest of the processing relies on. It also contains files with fit results (such that e.g. `04` can be run without `03`).

`01` and `02` start from the `.npy` exported MINFLUX data. For the code in this repository it is never necessary to download the `.msr` data.
