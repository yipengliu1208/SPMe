# SPMe

A Python implementation of an isothermal **single-particle model with electrolyte dynamics (SPMe)** for lithium-ion cell simulation. The model simulates terminal voltage from sampled total current and cell parameters.

The implementation keeps the physical equations and state-space matrices visible in separate source files:

- Solid-phase radial diffusion: finite difference method (FDM).
- Electrolyte transport across the negative electrode, separator and positive electrode: finite volume method (FVM).
- Concentration state updates driven by the supplied current profile.
- Reaction kinetics: symmetric Butler-Volmer (BV) equation.
- Voltage: electrode OCP difference, reaction overpotentials, electrolyte concentration contribution and electrolyte/solid ohmic losses.

Positive total current means **discharge**; negative current means charge. Current is in **A**, time in **s**, concentration in **mol/m^3**, and voltage in **V**.

## Installation

The model depends on NumPy and SciPy. It has been run with Python 3.11.

```bash
git clone https://github.com/yipengliu1208/SPMe.git
cd SPMe
python -m pip install numpy scipy
```

## Model interface

`main_vt(time, current, cell_parameters=parameters)` in [`main.py`](main.py) returns two arrays containing time [s] and terminal voltage [V].

The time and current arrays must have equal lengths, and time samples must be evenly spaced. The initial concentrations apply at `time[0]`. The input is held at `current[k]` over each interval `[time[k], time[k+1])`. The final current sample is used to calculate the final voltage without advancing the state.

The model checks voltage and concentration limits at each sample time. It returns the sampled point when a voltage limit is reached and ends before a sample with an out-of-range concentration.

## Model equations

After spatial discretization, each concentration subsystem has the form

$$
\dot{x}(t)=A_c x(t)+B_c I(t).
$$

For a constant current over a time interval $\Delta t$, the state update is

$$
x_{k+1}=A_d x_k+B_d I_k,\qquad
A_d=e^{A_c\Delta t},\qquad
B_d=\int_0^{\Delta t}e^{A_c\tau}B_c\,d\tau.
$$

The phase-model modules return the discrete state and input matrices. Particle surface concentrations are reconstructed through the solid-model output equation $c_{ss}=Cx+DI$.

Terminal voltage is assembled as

$$
V_t=(U_p-U_n)+(\eta_p-\eta_n)+\eta_c
+\phi_{e,\mathrm{ohmic}}+\phi_{s,\mathrm{ohmic}}.
$$

The ohmic and overpotential contributions retain their algebraic signs. Electrolyte concentration averages and averages of its square root are calculated from the electrolyte states for the voltage and BV closures.

## Source files

| File | Role |
| --- | --- |
| [`main.py`](main.py) | Voltage-only entry point, `main_vt`. |
| `solid_phase_model.py` | Radial FDM matrices, particle coordinates and surface-concentration output. |
| `electrolyte_model.py` | Three-region FVM matrices, electrolyte coordinates and regional averages. |
| `bv_overpotential.py` | Exchange current densities and symmetric BV inversion. |
| `cell_voltage.py` | Terminal-voltage contributions. |
| `parameters.py` | Example cell/material parameters, initial conditions and selected OCP functions. |
| `ocp.py` | Empirical graphite and lithium cobalt oxide OCP functions. |

To use another parameter set, create a module with the same required names and pass that module as `cell_parameters`. Set its `ocp_negative` and `ocp_positive` functions accordingly. Uniform initial concentrations are defined by `negative_c_init`, `positive_c_init` and `c_e0` in that module. Derived parameters must be recalculated consistently when changing the underlying values.

## Scope and references

This is an isothermal reduced electrochemical model. It does not include thermal dynamics, aging, side reactions or an additional series film-resistance term. The example transport and kinetic parameters are evaluated at the reference conditions documented in `parameters.py`; changing temperature alone does not reevaluate those material functions. Predictions depend on the parameter set and model assumptions and require validation for the target cell and operating conditions.

Parameter and OCP provenance is documented in the source, including the [PyBaMM v25.8.0 Marquis2019 parameter set](https://github.com/pybamm-team/PyBaMM/blob/v25.8.0/src/pybamm/input/parameters/lithium_ion/Marquis2019.py). The voltage implementation cites [Marquis et al. (2019), *An asymptotic derivation of a single particle model with electrolyte*](https://arxiv.org/abs/1905.12553).
