"""Define cell geometry, material properties, transport, kinetics and stop limits.
Positive total current means discharge. All parameters use SI units unless stated.

Variable reference:
    F: Faraday constant [C/mol]; R: universal gas constant [J/(mol K)].
    negative_* / positive_*: negative/positive electrode parameters.
        sign: reaction direction, +1 negative and -1 positive [-].
        L: electrode thickness [m].
        epsilon_s: active solid volume fraction [-].
        epsilon_e: electrolyte porosity [-].
        R_s: particle radius [m]; D_s: solid diffusivity [m^2/s].
        c_max: maximum solid concentration [mol/m^3].
        c_init: uniform initial solid concentration [mol/m^3].
        k: reaction rate constant [m^2.5/(mol^0.5 s)]; i0 uses F*k.
        sigma_eff: effective electronic conductivity [S/m], already corrected
            for porous structure.
        a_s: active interfacial area per electrode volume [1/m].
    Area: electrode cross-sectional area [m^2]; T: fixed temperature [K].
    L_sep: separator thickness [m]; epsilon_sep: separator porosity [-].
    c_e0: initial/reference electrolyte concentration [mol/m^3].
    t_plus: cation transference number [-]; brug: electrolyte Bruggeman exponent [-].
    negative_sigma / positive_sigma: intrinsic electronic conductivities [S/m].
    negative_brug_s / positive_brug_s: electrode Bruggeman exponents [-].
    L_e: regional thickness array [m], in negative/separator/positive order.
    epsilon_e: regional porosity array [-], in the same order.
    D_e_bulk / D_e_eff: bulk/regional effective electrolyte diffusivities [m^2/s].
    kappa_e_bulk / kappa_e_eff: bulk/regional effective ionic conductivities [S/m].
    nominal_capacity_Ah: reference capacity for C-rate conversion [A h].
    ocp_negative / ocp_positive: selected electrode OCP functions U(theta) [V].
    V_min / V_max: lower/upper terminal-voltage limits [V].

Material data:
Graphite/LCO example parameters from the PyBaMM v25.8.0 Marquis2019
dataset. Provenance labels on assignments distinguish data, evaluated functions,
derived values, conventions and numerical thresholds.
[M]: PyBaMM v25.8.0 Marquis2019 parameter dictionary or named material function.
[C]: physical constants, matching the PyBaMM/SciPy values at the shown precision.
[D]: derived from the indicated parameters and formula.
[S]: implementation convention or numerical threshold; not material data.

Sources:
https://github.com/pybamm-team/PyBaMM/blob/v25.8.0/src/pybamm/input/parameters/lithium_ion/Marquis2019.py
https://github.com/pybamm-team/PyBaMM/blob/v25.8.0/src/pybamm/models/submodels/transport_efficiency/bruggeman.py
https://github.com/pybamm-team/PyBaMM/blob/v25.8.0/src/pybamm/models/submodels/electrode/ohm/composite_ohm.py
https://github.com/pybamm-team/PyBaMM/blob/v25.8.0/src/pybamm/parameters/constants.py

Approximations:
D_s(T), D_e(c_e,T), conductivity and kinetic prefactors are evaluated at
298.15 K; electrolyte transport is also frozen at c_e0. Changing T alone
does not reevaluate these material functions. Bruggeman transport efficiency
uses epsilon_e**b for electrolyte and (1-epsilon_e)**b_s for the solid
conducting phase, following the cited PyBaMM implementation. The latter
includes all non-pore material and is not epsilon_s**b_s.
"""
import numpy as np
import ocp


# Physical constants.
F = 96485.33212  # [C] Faraday constant [C/mol].
R = 8.314462618  # [C] Gas constant [J/(mol K)].

# Negative electrode.
negative_sign = +1  # [S] Reaction sign for positive discharge current [-].
negative_L = 100e-6  # [M] Negative electrode thickness [m].
negative_epsilon_s = 0.6  # [M] Negative active material volume fraction [-].
negative_epsilon_e = 0.3  # [M] Negative electrode porosity [-].
negative_R_s = 10e-6  # [M] Negative particle radius [m].
negative_D_s = 3.9e-14  # [M evaluated] D_ref at 298.15 K; Arrhenius E_D=42770 J/mol [m^2/s].
negative_c_max = 24983.2619938437  # [M] Maximum negative solid concentration [mol/m^3].
negative_c_init = 19986.609595075  # [M] Initial negative solid concentration; uniform in r [mol/m^3].
negative_k = 2e-5 / F  # [D] 2e-5 is [M] m_ref at 298.15 K; k=m_ref/F, so i0 uses F*k.
negative_sigma = 100.0              # [M] Negative electrode conductivity [S/m], before porous correction.
negative_brug_s = 1.5               # [M] Negative electrode Bruggeman coefficient (electrode) [-].
negative_sigma_eff = negative_sigma * (1-negative_epsilon_e)**negative_brug_s  # [D] sigma * (1-porosity)**b_s, as in PyBaMM [S/m].
negative_a_s = 3.0 * negative_epsilon_s / negative_R_s  # [D] Spherical particles: surface/volume * epsilon_s = 3*epsilon_s/R_s [1/m].

# Positive electrode.
positive_sign = -1  # [S] Reaction sign for positive discharge current [-].
positive_L = 100e-6  # [M] Positive electrode thickness [m].
positive_epsilon_s = 0.5  # [M] Positive active material volume fraction [-].
positive_epsilon_e = 0.3  # [M] Positive electrode porosity [-].
positive_R_s = 10e-6  # [M] Positive particle radius [m].
positive_D_s = 1e-13  # [M evaluated] D_ref at 298.15 K; Arrhenius E_D=18550 J/mol [m^2/s].
positive_c_max = 51217.9257309275  # [M] Maximum positive solid concentration [mol/m^3].
positive_c_init = 30730.7554385565  # [M] Initial positive solid concentration; uniform in r [mol/m^3].
positive_k = 6e-7 / F  # [D] 6e-7 is [M] m_ref at 298.15 K; k=m_ref/F, so i0 uses F*k.
positive_sigma = 10.0               # [M] Positive electrode conductivity [S/m], before porous correction.
positive_brug_s = 1.5               # [M] Positive electrode Bruggeman coefficient (electrode) [-].
positive_sigma_eff = positive_sigma * (1-positive_epsilon_e)**positive_brug_s  # [D] sigma * (1-porosity)**b_s, as in PyBaMM [S/m].
positive_a_s = 3.0 * positive_epsilon_s / positive_R_s  # [D] Spherical particles: surface/volume * epsilon_s = 3*epsilon_s/R_s [1/m].

# Cell geometry, reference conditions and stopping thresholds.
Area = 0.137 * 0.207  # [D] [M] Electrode height * width [m^2]; total current is in A.
T = 298.15  # [M] Initial/reference temperature [K]; held fixed in this model.
L_sep = 25e-6  # [M] Separator thickness [m].
epsilon_sep = 1.0  # [M] Separator porosity [-].
c_e0 = 1000.0  # [M] Initial electrolyte concentration [mol/m^3]; also transport reference.
t_plus = 0.4  # [M] Cation transference number [-].
brug = 1.5  # [M] All three electrolyte Bruggeman exponents equal 1.5 [-].
V_min = 3.105  # [M] Lower voltage cut-off [V].
V_max = 4.1  # [M] Upper voltage cut-off [V].

# Regional geometry and porosity: negative electrode, separator, positive electrode.
L_e = np.array([negative_L, L_sep, positive_L])  # [D] Existing thicknesses in negative/separator/positive order [m].
epsilon_e = np.array([negative_epsilon_e, epsilon_sep, positive_epsilon_e])  # [D] Existing porosities in the same regional order [-].

# Electrolyte transport coefficients evaluated at the reference concentration.
D_e_bulk = 5.34e-10 * np.exp(-0.65 * c_e0 / 1000.0)  # [M evaluated] Capiglia1999 diffusivity at c_e0, 298.15 K [m^2/s].
D_e_eff = D_e_bulk * epsilon_e**brug  # [D] Bruggeman: D_e_bulk * epsilon_e**brug [m^2/s].
kappa_e_bulk = (  # [M evaluated] Capiglia1999 conductivity at c_e0, 298.15 K [S/m].
    0.0911
    + 1.9101*(c_e0/1000.0)
    - 1.052*(c_e0/1000.0)**2
    + 0.1554*(c_e0/1000.0)**3
)
kappa_e_eff = kappa_e_bulk * epsilon_e**brug  # [D] Bruggeman: kappa_e_bulk * epsilon_e**brug [S/m].

# Reference capacity: 1C current density of 24 A/m^2, multiplied by one hour.
nominal_capacity_Ah = 24.0 * Area  # [D] 24 A/m^2 * Area * 1 h = [M] 0.680616 Ah.


# Equilibrium potentials [V] as functions of dimensionless particle stoichiometry.
ocp_negative = ocp.ocp_negative  # [M] graphite_mcmb2528_ocp_Dualfoil, implemented in ocp.py.
ocp_positive = ocp.ocp_positive  # [M] lico2_ocp_Dualfoil, implemented in ocp.py.
