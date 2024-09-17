This directory contains the essential code used in the the Stasinski et al. 2024 paper.

TO run the dFIC tuning we recommend to read use and adapt the code in sample_dFIC_script.py:

- using your own SC would require changing the setup of connectivity object

This code requires a preinstalled The Virtual Brain and uses two main functions from:
-jansen_rit_FIC.py -> to run the tuning
-jansen_rit_postFIC.py  -> to run the post FIC simulations

Additionally dFIC_functions.py contains numerous helper functions used for plotting, FC and FCD calculations, Poincare Maps etc.

this directory contain also two numpy files:
 - all_ys_vs_mu array.npy - Determining the optimal initial conditions for any given mu
 - my_y0_dict.py - Determining the optimal mu for any given target

Last but not least we included the code used to run permutation testing used in our statistical analysis.
