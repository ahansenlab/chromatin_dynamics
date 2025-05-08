Data analysis of MINFLUX, SPT, SRLCI data
=========================================

- MINFLUX & SPT: raw data here
- units: MINFLUX m, SPT & SRLCI μm
- SRLCI: copied from (Gabriele, 2022), .h5 here
- python: 3.9.9
- bayesmsd has compiled cython only for python 3.9
- need at least about 8G RAM to hold SPT 2-locus data set
- our CPU: Intel(R) Xeon(R) Platinum 8360Y @ 2.40 GHz, 72 cores 


What does what?
---------------
01_L-sweep
- .msr --> .npy: in iMSPECTOR (v16.3.15620)
- .npy --> .h5: 00 (20240825_minflux_L-sweep_raw.h5)
- cleaning data: 01 (20250302_minflux_L-sweep_clean.h5)
- plots for Fig. 2: 02

02_data_processing/01_MINFLUX
- .msr --> .npy: in iMSPECTOR (v16.3.15620)
- .npy --> .h5: 00 (20250302_minflux_raw.h5)
- cleaning data
  + 01 for array (20250302_minflux_array_clean.h5)
  + 02 for H2B (20250302_minflux_H2B_clean.h5)
- identifying stuck trajectories in U2OS: 03 (20250302_single-traj_NPFit_H2B_clean.h5)

02_data_processing/02_SPT
- H2B, .xml --> .h5: 01 (20250121_SPT_H2B.h5)
- array, .xml --> .h5: 02 (20250411_SPT_array_CTCF.h5)

02_data_processing/03_assemble: merge data sets and make everything consistent
--> 20250411_chromatin_dynamics_all_data.h5 (1.8G)

02/data_processing/04_check_single-traj_MSDs.ipynb
--> plot single trajectory MSDs for all conditions

03_fitting
- mESC: 01 (`20250327_(fitres|mci)_NPFit-aGparam_mESC.h5`)
- U2OS: 02 (`20250327_(fitres|mci)_NPFit-aGparam_U2OS.h5`)
