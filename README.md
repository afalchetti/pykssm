# pykssm
*Copyright (c) 2016, Angelo Falchetti*

Simple and flexible library to do kernel state-space modelling (KSSM).

This technique allows the user to estimate the transition function of
a nonlinear state-space model without supervision, i.e. only by looking
at the output observations of the system, with no access to the internal
state; using an appropriate parametrization in the reproducing kernel
hilbert space.

Based on the work of Tobar, Djurić and Mandic, "Unsupervised State-Space
Modelling Using Reproducing Kernels", IEEE Transactions in Signal Processing,
2015.
If you've found the code useful for your own work, please cite the paper.
