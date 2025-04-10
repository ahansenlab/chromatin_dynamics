# First passage time (FPT) simulations

This directory contains the code used to simulate first passage times (FPTs)
for [TITLE]. The jupyter notebook contains the code required to simulate
fractional Brownian motion (fBm) trajectories, to calculate first passage times
and to determine parameters for which the FPT distribution converges.

## Summary of contents
This directory contains the following files:
- `all_first_passage_times.npy`: an .npy file containing the first passage times
calculated for the paper.
- `environment.yml`: a file specifying the dependencies required for the
simulations.
- `first_passage_time_simulations.ipynb`: a jupyter notebook containing the main
analysis code.
- `trajectories.py`: a python file containing the submodule for simulating the
fBms using the Davies-Harte algorithm.

## How to use
To run the python scripts provided, first set up a `conda` environment with the
packages specified in `environment.yml`. This can be done with the command

```
conda env create -f environment.yml
```

The environment can then be activated using

```
conda activate imaging_analysis
```

To make the environment available to jupyter notebooks, run the following commands
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=first_passage_times
```
and use the `first_passage_times` kernel.

Alternatively, the environment can be setup by installing the listed packages
from the necessary source by hand in a fresh conda environment. To run the
jupyter notebooks, use the `jupyter notebook` command in a conda environment
with jupyter installed. The jupyter notebook should be run from the directory
containing `trajectories.py`.
