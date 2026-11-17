# SPMe

A Python implementation of an isothermal **single-particle model with electrolyte dynamics (SPMe)** for lithium-ion cell simulation. Given sampled total current and a parameter module, the model computes concentration histories, reaction overpotentials and terminal voltage.

The implementation keeps the physical equations and state-space matrices visible in separate source files:

- Solid-phase radial diffusion: finite difference method (FDM).
- Electrolyte transport across the negative electrode, separator and positive electrode: finite volume method (FVM).
- Time propagation: zero-order-hold (ZOH) discretization using matrix exponentials.
- Reaction kinetics: symmetric Butler–Volmer (BV) equation.
- Voltage: electrode OCP difference, reaction overpotentials, electrolyte concentration contribution and electrolyte/solid ohmic losses.

Positive total current means **discharge**; negative current means charge. Current is in **A**, time in **s**, concentration in **mol/m³**, and voltage in **V**.

## Installation

The model depends on NumPy and SciPy. It has been run with Python 3.11.

```bash
git clone https://github.com/yipengliu1208/SPMe.git
cd SPMe
python -m pip install numpy scipy
```
## Inputs and time convention

```python
main_spme(time, current, cell_parameters,
          n_s=60, n_e=(30, 10, 30), output_dt=1, voltage_limits=True)
```

| Argument | Meaning |
| --- | --- |
| `time` | Sample times [s]. Initial conditions apply at `time[0]`. |
| `current` | Sampled total cell current [A]. |
| `cell_parameters` | Python module containing geometry, material parameters, initial concentrations, OCP functions and voltage limits. |
| `n_s` | Number of internal radial solid states per electrode. |
| `n_e` | FVM cell counts in negative/separator/positive order. |
| `output_dt` | Maximum output spacing [s]. Each input interval receives an evenly spaced output grid. |
| `voltage_limits` | Enable voltage cutoffs. Physical concentration limits always apply. |

Over each interval `[time[k], time[k+1])`, the input is held at `current[k]`. The final current sample determines the algebraic outputs at the final time without advancing the state further. At current switches, the returned sample uses the new current (the right limit), provided that operating point is admissible. If the new current immediately violates a limit, simulation stops and retains the preceding valid result.

Output times may include extra points and a located cutoff time. Use `result["time"]` with the returned data, rather than assuming the output has the same length as the input profile.

## Model equations

After spatial discretization, each concentration subsystem has the form

$$
\dot{x}(t)=A_c x(t)+B_c I(t).
$$

For a constant current over a time interval $\Delta t$, ZOH gives

$$
x_{k+1}=A_d x_k+B_d I_k,\qquad
A_d=e^{A_c\Delta t},\qquad
B_d=\int_0^{\Delta t}e^{A_c\tau}B_c\,d\tau.
$$

The phase-model modules compute these matrices using an augmented matrix exponential. Particle surface concentrations are reconstructed through the solid-model output equation $c_{ss}=Cx+DI$.

Terminal voltage is assembled as

$$
V_t=(U_p-U_n)+(\eta_p-\eta_n)+\eta_c
+\phi_{e,\mathrm{ohmic}}+\phi_{s,\mathrm{ohmic}}.
$$

The ohmic and overpotential contributions retain their algebraic signs. Electrolyte concentration averages and averages of its square root are calculated from the FVM states for the voltage and BV closures.

## Outputs

The returned dictionary includes:

| Keys | Description |
| --- | --- |
| `time`, `current`, `segment` | Output times, total current and input-interval indices. |
| `V` | Terminal voltage [V]. |
| `U_n`, `U_p`, `U_eq` | Negative/positive OCP and their difference [V]. |
| `eta_n`, `eta_p`, `eta_r` | Electrode reaction overpotentials and `eta_p - eta_n` [V]. |
| `eta_c` | Electrolyte concentration voltage contribution [V]. |
| `phi_e_ohmic`, `phi_s_ohmic` | Electrolyte and solid ohmic voltage contributions [V]. |
| `i0_n`, `i0_p` | Averaged exchange current densities [A/m²]. |
| `c_s_n`, `c_s_p` | Internal solid concentration histories, shape `(n_s, Nt)`. |
| `c_ss_n`, `c_ss_p` | Particle surface concentration histories, shape `(Nt,)`. |
| `c_e` | Electrolyte cell-average concentrations, shape `(sum(n_e), Nt)`. |
| `ce_average_n`, `ce_average_sep`, `ce_average_p` | Regional electrolyte concentration averages. |
| `ce_sqrt_average_n`, `ce_sqrt_average_p` | Electrode averages of `sqrt(c_e)`. |
| `X` | Combined state history: negative solid, positive solid, then electrolyte. |
| `r_n`, `r_p`, `x_e` | Internal particle radii and electrolyte cell-center coordinates [m]. |
| `stop_reason` | Completion or cutoff description. |

Here `Nt` is the number of returned time samples; concentration arrays have time along columns. Internal particle states exclude the center and surface; surface concentrations are returned separately.

The driver checks limits at intervals no greater than 1 s and locates bracketed crossings by root finding. At a physical concentration cutoff, the true endpoint state is retained and its voltage/kinetic outputs are `NaN`, because the BV expression is undefined at that boundary.

## Source files

| File | Role |
| --- | --- |
| `main_spme.py` | Callable simulation driver, initial states, ZOH propagation, cutoff handling and output assembly. |
| `solid_phase_model.py` | Radial FDM matrices, surface-concentration output and ZOH discretization. |
| `electrolyte_model.py` | Three-region FVM matrices, spatial outputs/averages and ZOH discretization. |
| `bv_overpotential.py` | Exchange current densities and symmetric BV inversion. |
| `cell_voltage.py` | Terminal-voltage contributions. |
| `parameters.py` | Example cell/material parameters, initial conditions and selected OCP functions. |
| `ocp.py` | Empirical graphite and lithium cobalt oxide OCP functions. |

To use another parameter set, create a module with the same required names and pass that module as `cell_parameters`. Set its `ocp_negative` and `ocp_positive` functions accordingly. Uniform initial concentrations are defined by `negative_c_init`, `positive_c_init` and `c_e0` in that module. Derived parameters must be recalculated consistently when changing the underlying values.

## Scope and references

This is an isothermal reduced electrochemical model. It does not include thermal dynamics, aging, side reactions or an additional series film-resistance term. The example transport and kinetic parameters are evaluated at the reference conditions documented in `parameters.py`; changing temperature alone does not reevaluate those material functions. Predictions depend on the parameter set and model assumptions and require validation for the target cell and operating conditions.

Parameter and OCP provenance is documented in the source, including the [PyBaMM v25.8.0 Marquis2019 parameter set](https://github.com/pybamm-team/PyBaMM/blob/v25.8.0/src/pybamm/input/parameters/lithium_ion/Marquis2019.py). The voltage implementation cites [Marquis et al. (2019), *An asymptotic derivation of a single particle model with electrolyte*](https://arxiv.org/abs/1905.12553).
