"""SPMe: solid FDM, electrolyte FVM.
Positive total current means discharge. Concentrations use mol/m^3; time uses seconds.

Inputs:
    cell_parameters: selected module of cell/material constants and electrode OCP functions.
    time / current: equal-length 1D sample arrays [s] and [A]; time must increase strictly.
        The initial state belongs to time[0]. ZOH holds current[k] until time[k+1].
        The final current sample is used for the final algebraic output, without extra duration.
        Outputs retain the right limit at current switches; additional output/cutoff points may occur.
    n_s: internal solid nodes per electrode; n_e: negative/separator/positive FVM cell counts.
    output_dt: maximum output spacing [s]; each segment has an evenly spaced output grid.
    voltage_limits: enable voltage cutoffs; concentration cutoffs always apply.

Initial conditions and internal settings:
    cell_parameters.negative_c_init / positive_c_init / c_e0: uniform initial concentrations.
    max_step: maximum cutoff-check interval [s], assigned inside main_spme.
    event_tol: cutoff root time tolerance [s], assigned inside main_spme.

Returned dictionary:
    time / current / segment: output times [s], total currents [A] and interval indices.
    At a physical concentration cutoff the true endpoint is retained; voltage and
    kinetic outputs there are NaN because BV is undefined at the boundary.
    X: full concentration history; rows are negative solid, positive solid, then electrolyte.
    c_s_n / c_s_p / c_e: corresponding concentration arrays; columns are time samples.
    r_n / r_p / x_e: internal particle radii and electrolyte cell centers [m].
    V / U_n / U_p / U_eq / eta_r / eta_c / phi_e_ohmic / phi_s_ohmic / eta_n / eta_p: [V].
    i0_n / i0_p: averaged exchange current densities [A/m^2].
    c_ss_n / c_ss_p: particle surface concentrations.
    ce_average_n / ce_average_sep / ce_average_p: regional mean electrolyte concentrations.
    ce_sqrt_average_n / ce_sqrt_average_p: mean sqrt(ce) [(mol/m^3)^0.5].
    stop_reason: termination label.

Internal variables:
    input_time / input_current: original sample arrays; index: sample-interval index.
    valid: output-domain mask; voltage: valid voltage and kinetic output histories.
    zoh_matrices: cache mapping propagation intervals [s] to discrete A and B.
    discrete: assembled ZOH dictionary for interval dt [s].
    solid_n_discrete / solid_p_discrete / electrolyte_discrete: discrete submodels.
    State update: x_next=A_d@x+B_d[:,0]*I; output matrices C and D are unchanged.
    solid_n_inputs / solid_p_inputs: named particle parameters reused for each dt.
    Submodel A/B are discrete; A_c/B_c retain the continuous reference matrices.

"""
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import brentq
from solid_phase_model import build_solid_model
from electrolyte_model import build_electrolyte_model
from cell_voltage import cell_voltage


def main_spme(time, current, cell_parameters, n_s=60, n_e=(30, 10, 30), output_dt=1, voltage_limits=True):
    max_step = 1.0       # Maximum cutoff-check interval [s].
    event_tol = 1e-8     # Absolute cutoff root time tolerance [s].

    # Validate sampled inputs, output spacing and initial concentrations.
    input_time = np.asarray(time, dtype=float)
    input_current = np.asarray(current, dtype=float)
    if (input_time.ndim != 1 or input_current.ndim != 1 or len(input_time) < 2
            or len(input_time) != len(input_current)
            or not np.all(np.isfinite(input_time)) or not np.all(np.isfinite(input_current))
            or not np.all(np.isfinite(np.diff(input_time)))
            or np.any(np.diff(input_time) <= 0)):
        raise ValueError("Provide equal-length finite 1D time/current arrays and strictly increasing times.")
    if not np.isfinite(output_dt) or output_dt <= 0:
        raise ValueError("output_dt must be positive and finite.")
    if not np.all(np.isfinite([cell_parameters.negative_c_init,
                               cell_parameters.positive_c_init, cell_parameters.c_e0])):
        raise ValueError("Initial concentrations must be finite scalars.")

    # Build solid-phase and electrolyte matrices and initial states.
    if not (cell_parameters.Area > 0 and cell_parameters.T > 0 and cell_parameters.c_e0 > 0 and cell_parameters.L_sep > 0):
        raise ValueError("Area, T, c_e0 and L_sep must be positive.")
    if not (0 < cell_parameters.epsilon_sep <= 1 and 0 <= cell_parameters.t_plus < 1):
        raise ValueError("Invalid separator porosity or transference number.")
    if not cell_parameters.V_min < cell_parameters.V_max:
        raise ValueError("V_min must be below V_max.")

    # Negative electrode.
    if cell_parameters.negative_sign != +1 or min(cell_parameters.negative_L, cell_parameters.negative_R_s, cell_parameters.negative_D_s,
                                 cell_parameters.negative_c_max, cell_parameters.negative_k, cell_parameters.negative_sigma_eff) <= 0:
        raise ValueError("Invalid negative electrode parameters.")
    if not (0 < cell_parameters.negative_epsilon_s <= 1 and 0 < cell_parameters.negative_epsilon_e <= 1
            and cell_parameters.negative_epsilon_s + cell_parameters.negative_epsilon_e <= 1):
        raise ValueError("Invalid negative volume fractions.")
    if not 0 < cell_parameters.negative_c_init/cell_parameters.negative_c_max < 1:
        raise ValueError("Invalid negative initial concentration.")

    # Positive electrode.
    if cell_parameters.positive_sign != -1 or min(cell_parameters.positive_L, cell_parameters.positive_R_s, cell_parameters.positive_D_s,
                                 cell_parameters.positive_c_max, cell_parameters.positive_k, cell_parameters.positive_sigma_eff) <= 0:
        raise ValueError("Invalid positive electrode parameters.")
    if not (0 < cell_parameters.positive_epsilon_s <= 1 and 0 < cell_parameters.positive_epsilon_e <= 1
            and cell_parameters.positive_epsilon_s + cell_parameters.positive_epsilon_e <= 1):
        raise ValueError("Invalid positive volume fractions.")
    if not 0 < cell_parameters.positive_c_init/cell_parameters.positive_c_max < 1:
        raise ValueError("Invalid positive initial concentration.")
    if not callable(cell_parameters.ocp_negative) or not callable(cell_parameters.ocp_positive):
        raise ValueError("The selected parameter module must provide two callable OCP functions.")
    solid_n_inputs = dict(
        R_s=cell_parameters.negative_R_s,
        D_s=cell_parameters.negative_D_s,
        epsilon_s=cell_parameters.negative_epsilon_s,
        L=cell_parameters.negative_L,
        sign=cell_parameters.negative_sign,
        Area=cell_parameters.Area,
        F=cell_parameters.F,
        n=n_s,
    )
    solid_p_inputs = dict(
        R_s=cell_parameters.positive_R_s,
        D_s=cell_parameters.positive_D_s,
        epsilon_s=cell_parameters.positive_epsilon_s,
        L=cell_parameters.positive_L,
        sign=cell_parameters.positive_sign,
        Area=cell_parameters.Area,
        F=cell_parameters.F,
        n=n_s,
    )
    # Zero interval initializes the mesh and output matrices.
    solid_n = build_solid_model(**solid_n_inputs, dt=0.0)
    solid_p = build_solid_model(**solid_p_inputs, dt=0.0)
    electrolyte = build_electrolyte_model(cell_parameters, n_e, dt=0.0)

    # Initial concentration states of the three subsystems sharing I.
    Ne = sum(n_e)
    x0 = np.r_[np.full(n_s, cell_parameters.negative_c_init),
               np.full(n_s, cell_parameters.positive_c_init), np.full(Ne, cell_parameters.c_e0)]
    sn = slice(0, n_s)
    sp = slice(n_s, 2*n_s)
    se = slice(2*n_s, 2*n_s+Ne)

    # Repeated concentration and cutoff calculations.
    def concentration_outputs(X, current):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        I = np.broadcast_to(np.asarray(current), (X.shape[1],))[None, :]
        c_ss_n = (solid_n.C @ X[sn] + solid_n.D @ I)[0]
        c_ss_p = (solid_p.C @ X[sp] + solid_p.D @ I)[0]
        ce = X[se]
        return c_ss_n, c_ss_p, ce

    # Assemble the discrete submodels for each propagation interval [s].
    def discrete_matrices(dt):
        """Assemble ZOH matrices returned by the two physical model modules."""
        solid_n_discrete = build_solid_model(**solid_n_inputs, dt=dt)
        solid_p_discrete = build_solid_model(**solid_p_inputs, dt=dt)
        electrolyte_discrete = build_electrolyte_model(cell_parameters, n_e, dt=dt)
        return {
            "A": block_diag(solid_n_discrete.A, solid_p_discrete.A, electrolyte_discrete.A),
            "B": np.vstack([solid_n_discrete.B, solid_p_discrete.B, electrolyte_discrete.B]),
        }

    zoh_matrices = {}

    def cutoff_margins(x, current):
        concentration_values = concentration_outputs(x, current)
        c_ss_n = concentration_values[0]
        c_ss_p = concentration_values[1]
        ce = concentration_values[2]
        theta_n = np.r_[x[sn], c_ss_n] / cell_parameters.negative_c_max
        theta_p = np.r_[x[sp], c_ss_p] / cell_parameters.positive_c_max
        theta = np.r_[theta_n, theta_p]
        margins = [float(min(theta.min(),
                         1-theta.max(),
                         ce.min()/cell_parameters.c_e0))]
        if voltage_limits:
            # Bound only voltage-trial inputs beyond physical limits; returned states are unchanged.
            c_ss_n = np.clip(c_ss_n, np.finfo(float).eps*cell_parameters.negative_c_max, (1-np.finfo(float).eps)*cell_parameters.negative_c_max)
            c_ss_p = np.clip(c_ss_p, np.finfo(float).eps*cell_parameters.positive_c_max, (1-np.finfo(float).eps)*cell_parameters.positive_c_max)
            ce = np.maximum(ce, np.finfo(float).eps*cell_parameters.c_e0)
            voltage = float(cell_voltage(
                current, c_ss_n, c_ss_p,
                electrolyte.C_average @ ce,
                electrolyte.C_average @ np.sqrt(ce),
                cell_parameters=cell_parameters,
            )["V"][0])
            margins.extend([voltage-cell_parameters.V_min, cell_parameters.V_max-voltage])
        return np.asarray(margins)

    # Advance each sample interval with ZOH current and retain right-limit outputs.
    t_start = float(input_time[0])
    state = x0.copy()
    all_t = []
    all_i = []
    all_x = []
    all_segment = []
    stop_reason = "protocol completed"
    event_names = ["concentration limit"]
    if voltage_limits:
        event_names += ["lower voltage cutoff", "upper voltage cutoff"]

    for index in range(len(input_time)-1):
        duration = input_time[index+1] - input_time[index]
        I = input_current[index]
        margins = cutoff_margins(state, I)
        if margins[0] <= 0:
            stop_reason = f"current step violates concentration limits (segment {index})"
            break
        if voltage_limits and not np.all(margins[1:] > 0):
            V_start = margins[1] + cell_parameters.V_min
            stop_reason = f"current step violates voltage limits (segment {index}, V={V_start:.6g})"
            break

        times = np.linspace(t_start, t_start+duration,
                            int(np.ceil(duration/output_dt))+1)
        sample_step = duration / (len(times)-1)
        substeps = int(np.ceil(sample_step/max_step))
        step = sample_step / substeps
        if step not in zoh_matrices:
            zoh_matrices[step] = discrete_matrices(step)
        A_d = zoh_matrices[step]["A"]
        B_d = zoh_matrices[step]["B"]
        ts = [times[0]]
        xs = [state.copy()]
        cutoff_reached = False

        for sample_index in range(1, len(times)):
            for substep_index in range(substeps):
                next_state = A_d @ state + B_d[:, 0]*I
                next_margins = cutoff_margins(next_state, I)
                crossings = np.flatnonzero((margins > 0) & (next_margins <= 0))
                if len(crossings):
                    # Locate every bracketed threshold and retain the earliest crossing.
                    roots = []
                    for event_index in crossings:
                        def event_value(tau):
                            discrete = discrete_matrices(tau)
                            x = discrete["A"] @ state + discrete["B"][:, 0]*I
                            return cutoff_margins(x, I)[event_index]

                        roots.append(brentq(event_value, 0.0, step, xtol=event_tol))
                    candidate_index = int(np.argmin(roots))
                    event_index = crossings[candidate_index]
                    tau = roots[candidate_index]
                    discrete = discrete_matrices(tau)
                    xe = discrete["A"] @ state + discrete["B"][:, 0]*I
                    te = times[sample_index-1] + substep_index*step + tau
                    if not np.isclose(ts[-1], te, rtol=0, atol=1e-10):
                        ts.append(te)
                        xs.append(xe)
                    else:
                        ts[-1] = te
                        xs[-1] = xe
                    stop_reason = event_names[event_index]
                    cutoff_reached = True
                    break
                state = next_state
                margins = next_margins
            if cutoff_reached:
                break
            ts.append(times[sample_index])
            xs.append(state.copy())

        ts = np.asarray(ts)
        xs = np.column_stack(xs)
        # Replace the preceding interval's endpoint with this admissible right limit.
        if all_t:
            all_t[-1] = all_t[-1][:-1]
            all_i[-1] = all_i[-1][:-1]
            all_x[-1] = all_x[-1][:, :-1]
            all_segment[-1] = all_segment[-1][:-1]
        all_t.append(ts)
        all_i.append(np.full(len(ts), I))
        all_x.append(xs)
        all_segment.append(np.full(len(ts), index, dtype=int))
        state = xs[:, -1]
        t_start = ts[-1]
        if cutoff_reached:
            break

    # The final sample changes the algebraic output only, with no extra propagation.
    if all_t and stop_reason == "protocol completed":
        all_i[-1][-1] = input_current[-1]
        margins = cutoff_margins(state, input_current[-1])
        if margins[0] <= 0:
            stop_reason = "concentration limit at final current sample"
        elif voltage_limits and not np.all(margins[1:] > 0):
            stop_reason = "voltage limit at final current sample"

    if not all_t:
        raise ValueError(f"No admissible initial operating point: {stop_reason}")
    time = np.concatenate(all_t)
    current = np.concatenate(all_i)
    X = np.hstack(all_x)
    segment = np.concatenate(all_segment)

    # Calculate physical output histories.
    concentration_values = concentration_outputs(X, current)
    c_ss_n = concentration_values[0]
    c_ss_p = concentration_values[1]
    ce = concentration_values[2]
    valid = np.ones(len(time), dtype=bool)
    if stop_reason.startswith("concentration limit"):
        valid[-1] = False
    if np.any(~np.isfinite(X)) or np.any(ce[:, valid] <= 0):
        raise ValueError("Nonfinite states or nonpositive electrolyte concentration.")
    # Piecewise-constant reconstruction of each FVM cell for spatial integrals.
    ce_average = electrolyte.C_average @ ce
    ce_sqrt_average = electrolyte.C_average @ np.sqrt(
        ce, out=np.full_like(ce, np.nan), where=ce >= 0)
    # BV is undefined at concentration endpoints. Retain the true boundary state
    # and mark its voltage/kinetic outputs as NaN, rather than altering concentration.
    voltage = cell_voltage(current[valid], c_ss_n[valid], c_ss_p[valid],
                           ce_average[:, valid], ce_sqrt_average[:, valid], cell_parameters=cell_parameters)
    result = {}
    for name, values in voltage.items():
        result[name] = np.full(len(time), np.nan)
        result[name][valid] = values
    result["c_ss_n"] = c_ss_n
    result["c_ss_p"] = c_ss_p
    result["ce_average_n"] = ce_average[0]
    result["ce_average_sep"] = ce_average[1]
    result["ce_average_p"] = ce_average[2]
    result["ce_sqrt_average_n"] = ce_sqrt_average[0]
    result["ce_sqrt_average_p"] = ce_sqrt_average[2]

    # Return data for external analysis and file output.
    result["time"] = time
    result["current"] = current
    result["X"] = X
    result["segment"] = segment
    result["c_s_n"] = X[sn]
    result["c_s_p"] = X[sp]
    result["c_e"] = ce
    result["r_n"] = solid_n.r
    result["r_p"] = solid_p.r
    result["x_e"] = electrolyte.centers
    result["stop_reason"] = stop_reason
    return result
