"""Evaluate material open-circuit equilibrium potentials from surface stoichiometry.

Reference:
Material parameter values and the empirical OCP formulas in parameters.py / ocp.py
are adapted from PyBaMM v25.8.0 Marquis2019.py:
https://github.com/pybamm-team/PyBaMM/blob/v25.8.0/src/pybamm/input/parameters/lithium_ion/Marquis2019.py
Source file blob SHA: c5c7523658cb20b1d466bbb64c0deedd862da7f9


"""
import numpy as np


def ocp_negative(theta):
    x = np.asarray(theta)
    return (0.194 + 1.5*np.exp(-120*x)
            + 0.0351*np.tanh((x-0.286)/0.083)
            - 0.0045*np.tanh((x-0.849)/0.119)
            - 0.035*np.tanh((x-0.9233)/0.05)
            - 0.0147*np.tanh((x-0.5)/0.034)
            - 0.102*np.tanh((x-0.194)/0.142)
            - 0.022*np.tanh((x-0.9)/0.0164)
            - 0.011*np.tanh((x-0.124)/0.0226)
            + 0.0155*np.tanh((x-0.105)/0.029))


def ocp_positive(theta):
    x = 1.062 * np.asarray(theta)
    return (2.16216 + 0.07645*np.tanh(30.834-54.4806*x)
            + 2.1581*np.tanh(52.294-50.294*x)
            - 0.14169*np.tanh(11.0923-19.8543*x)
            + 0.2051*np.tanh(1.4684-5.4888*x)
            + 0.2531*np.tanh((-x+0.56478)/0.1316)
            - 0.02167*np.tanh((x-0.525)/0.006))
