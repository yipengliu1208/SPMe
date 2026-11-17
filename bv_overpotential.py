"""Compute exchange current densities and reaction overpotentials.
Convention: n means negative; the p suffix means positive.
Positive total current means discharge.

Variable reference (SI units; physical inputs may be scalars or NumPy arrays
on the same time grid):
    cell_parameters: module containing cell and electrode cell_parameters.
    current: total current I [A].
    c_ss_n / c_ss_p: representative-particle surface concentrations [mol/m^3].
    mean_sqrt_ce_n / mean_sqrt_ce_p: electrode-thickness averages of sqrt(c_e)
        [(mol/m^3)^0.5], defined as integral(sqrt(c_e) dx)/L.
        Take the square root before averaging.
    cell_parameters.F: Faraday constant [C/mol]; cell_parameters.R: gas constant [J/(mol K)].
    cell_parameters.T: fixed temperature T [K]; cell_parameters.Area: electrode cross-sectional area [m^2].
    cell_parameters.negative_k / cell_parameters.positive_k: reaction rate constants [m^2.5/(mol^0.5 s)].
    cell_parameters.negative_c_max / cell_parameters.positive_c_max: maximum solid concentrations [mol/m^3].
    cell_parameters.negative_a_s / cell_parameters.positive_a_s: active interfacial area per electrode volume [1/m],
        a_s=3*epsilon_s/R_s.
    cell_parameters.negative_L / cell_parameters.positive_L: electrode thicknesses [m].
    i0_n / i0_p: averaged exchange current densities [A/m^2 of active interface].
    eta_n / eta_p: averaged negative/positive reaction overpotentials [V].
The returned dictionary uses the four keys i0_n, i0_p, eta_n and eta_p.

Equations:
    i0_bar = F*k*sqrt(c_ss*(c_max-c_ss))*mean_sqrt_ce
    eta_n = +(2*R*T/F)*asinh(I/(2*Area*a_s_n*L_n*i0_n))
    eta_p = -(2*R*T/F)*asinh(I/(2*Area*a_s_p*L_p*i0_p))
These equations use the selected SPMe averaged kinetic closure.

Model assumptions:
Symmetric transfer coefficients alpha_a=alpha_c=0.5 and an
electrode-averaged kinetic closure. The closure is a model approximation;
nonlinear BV evaluation and spatial averaging do not generally commute.
"""
import numpy as np


def bv_overpotential(current, c_ss_n, c_ss_p, mean_sqrt_ce_n, mean_sqrt_ce_p, cell_parameters):
    
    # Checking
    if (np.any(~np.isfinite(c_ss_n))
            or np.any((c_ss_n <= 0) | (c_ss_n >= cell_parameters.negative_c_max))):
        raise ValueError("negative surface concentration outside (0,c_max).")
    if (np.any(~np.isfinite(c_ss_p))
            or np.any((c_ss_p <= 0) | (c_ss_p >= cell_parameters.positive_c_max))):
        raise ValueError("positive surface concentration outside (0,c_max).")
    if np.any(~np.isfinite(mean_sqrt_ce_n)) or np.any(mean_sqrt_ce_n <= 0):
        raise ValueError("Negative mean sqrt electrolyte concentration must be positive.")
    if np.any(~np.isfinite(mean_sqrt_ce_p)) or np.any(mean_sqrt_ce_p <= 0):
        raise ValueError("Positive mean sqrt electrolyte concentration must be positive.")

    # Electrode-averaged exchange current densities
    i0_n = (cell_parameters.F*cell_parameters.negative_k
            * np.sqrt(c_ss_n*(cell_parameters.negative_c_max-c_ss_n)) * mean_sqrt_ce_n)
    i0_p = (cell_parameters.F*cell_parameters.positive_k
            * np.sqrt(c_ss_p*(cell_parameters.positive_c_max-c_ss_p)) * mean_sqrt_ce_p)

    # Inverse BV
    eta_n = 2*cell_parameters.R*cell_parameters.T/cell_parameters.F * np.arcsinh(current/(2*cell_parameters.Area*cell_parameters.negative_a_s*cell_parameters.negative_L*i0_n))
    eta_p = -2*cell_parameters.R*cell_parameters.T/cell_parameters.F * np.arcsinh(current/(2*cell_parameters.Area*cell_parameters.positive_a_s*cell_parameters.positive_L*i0_p))
    
    
    return {
        "i0_n": i0_n,
        "i0_p": i0_p,
        "eta_n": eta_n,
        "eta_p": eta_p,
    }
