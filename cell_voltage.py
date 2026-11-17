"""Compute terminal voltage from OCP, BV, electrolyte concentration and Ohmic terms.
Convention: regional arrays use negative/separator/positive order.
Positive total current means discharge; voltage contributions retain their signs.

Variable reference:
    cell_parameters: cell-parameter module; current: total current I [A].
    c_ss_n / c_ss_p: particle surface concentrations [mol/m^3],
        scalars or arrays of shape (Nt,).
    ce_average: regional thickness-averaged c_e [mol/m^3], shape (3,Nt) or (3,).
    ce_sqrt_average: regional averages of sqrt(c_e) [(mol/m^3)^0.5], same shape.
    Nt: number of time samples; regional indices 0/1/2 mean negative/separator/positive.
    cell_parameters.F: Faraday constant [C/mol]; cell_parameters.R: gas constant [J/(mol K)].
    cell_parameters.T: temperature [K]; cell_parameters.Area: cross-sectional area [m^2].
    cell_parameters.t_plus: cation transference number [-].
    cell_parameters.c_e0: initial/reference electrolyte concentration [mol/m^3].
    cell_parameters.negative_L / cell_parameters.positive_L: electrode thicknesses [m].
    cell_parameters.negative_sigma_eff / cell_parameters.positive_sigma_eff:
        effective electronic conductivities [S/m].
    cell_parameters.L_sep: separator thickness [m].
    U_n / U_p: electrode OCPs evaluated at surface stoichiometry [V].
    U_eq: U_p-U_n, the OCP difference [V].
    kinetics: BV output dictionary; i0_n/i0_p [A/m^2], eta_n/eta_p [V].
    eta_r: eta_p-eta_n, the combined reaction contribution [V].
    eta_c: electrode-averaged electrolyte concentration contribution [V].
    cell_parameters.kappa_e_eff: effective ionic conductivities [S/m],
        ordered negative/separator/positive.
    phi_e_ohmic: electrolyte Ohmic contribution [V], computed directly from
        -I/Area*(L_n/(3*kappa_n)+L_sep/kappa_sep+L_p/(3*kappa_p)).
    phi_s_ohmic: solid Ohmic contribution [V], computed directly from
        -I/(3*Area)*(L_n/sigma_n+L_p/sigma_p).
    V: terminal voltage Vt [V].
The returned dictionary contains these voltage quantities and the four BV outputs.

Equation: V = U_eq + eta_r + eta_c + phi_e_ohmic + phi_s_ohmic

Model assumptions:
Isothermal SPMe with electrode-averaged reaction and electrolyte
contributions, a linearized concentration contribution, and finite
ionic/electronic conductivities.
Reference: Marquis et al. (2019), equations (48j)-(48q),
https://arxiv.org/pdf/1905.12553v2
"""
from bv_overpotential import bv_overpotential


def cell_voltage(current, c_ss_n, c_ss_p, ce_average, ce_sqrt_average, cell_parameters):
    
    # OCV
    U_n = cell_parameters.ocp_negative(c_ss_n/cell_parameters.negative_c_max)
    U_p = cell_parameters.ocp_positive(c_ss_p/cell_parameters.positive_c_max)
    
    # Overpotential
    kinetics = bv_overpotential(current, c_ss_n, c_ss_p,
                                ce_sqrt_average[0], ce_sqrt_average[2], cell_parameters=cell_parameters)
    U_eq = U_p-U_n
    eta_r = kinetics["eta_p"] - kinetics["eta_n"]
    eta_c = (2*cell_parameters.R*cell_parameters.T/cell_parameters.F * (1-cell_parameters.t_plus)
             * (ce_average[2]-ce_average[0])/cell_parameters.c_e0)

    # Electrolyte ohmic drop
    phi_e_ohmic = -current/cell_parameters.Area * (
        cell_parameters.negative_L / (3*cell_parameters.kappa_e_eff[0])
        + cell_parameters.L_sep / cell_parameters.kappa_e_eff[1]
        + cell_parameters.positive_L / (3*cell_parameters.kappa_e_eff[2]))
    
    # Solid ohmic drop
    phi_s_ohmic = -current/(3*cell_parameters.Area) * (
        cell_parameters.negative_L / cell_parameters.negative_sigma_eff
        + cell_parameters.positive_L / cell_parameters.positive_sigma_eff)
    
    # Final Vt
    V = U_eq + eta_r + eta_c + phi_e_ohmic + phi_s_ohmic

    return {
        "V": V,
        "U_n": U_n,
        "U_p": U_p,
        "U_eq": U_eq,
        "eta_r": eta_r,
        "eta_c": eta_c,
        "phi_e_ohmic": phi_e_ohmic,
        "phi_s_ohmic": phi_s_ohmic,
        "i0_n": kinetics["i0_n"],
        "i0_p": kinetics["i0_p"],
        "eta_n": kinetics["eta_n"],
        "eta_p": kinetics["eta_p"],
    }
