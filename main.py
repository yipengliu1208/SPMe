"""Run the constant-transport SPMe and return time [s] and terminal voltage [V]."""

import numpy as np

import parameters
from cell_voltage import cell_voltage
from electrolyte_model import build_electrolyte_model
from solid_phase_model import build_solid_model


def main_vt(time, current, cell_parameters=parameters):
    """Use a uniform time grid and discharge-positive current samples in amperes."""
    time = np.asarray(time, dtype=float)
    current = np.asarray(current, dtype=float)
    if time.ndim != 1 or time.size < 2 or current.shape != time.shape:
        raise ValueError("Provide matching one-dimensional time and current arrays.")
    steps = np.diff(time)
    dt = steps[0]
    if (not np.isfinite(time).all() or not np.isfinite(current).all()
            or np.any(steps <= 0) or not np.allclose(steps, dt, rtol=1e-9, atol=1e-12)):
        raise ValueError("Inputs must be finite with positive, uniform time spacing.")

    p = cell_parameters
    n_s = 60
    n_e = (30, 10, 30)
    negative = build_solid_model(p.negative_R_s, p.negative_D_s, p.negative_epsilon_s,
                                 p.negative_L, p.negative_sign, p.Area, n_s, dt=dt, F=p.F)
    positive = build_solid_model(p.positive_R_s, p.positive_D_s, p.positive_epsilon_s,
                                 p.positive_L, p.positive_sign, p.Area, n_s, dt=dt, F=p.F)
    electrolyte = build_electrolyte_model(p, n_e, dt=dt)

    # Keep only the current concentration state of each subsystem.
    c_s_n = np.full(n_s, p.negative_c_init)
    c_s_p = np.full(n_s, p.positive_c_init)
    c_e = np.full(sum(n_e), p.c_e0)
    voltage = np.empty(time.size)

    for k, I in enumerate(current):
        c_ss_n = (negative.C @ c_s_n)[0] + negative.D[0, 0]*I
        c_ss_p = (positive.C @ c_s_p)[0] + positive.D[0, 0]*I
        if (min(c_s_n.min(), c_ss_n) <= 0 or max(c_s_n.max(), c_ss_n) >= p.negative_c_max
                or min(c_s_p.min(), c_ss_p) <= 0 or max(c_s_p.max(), c_ss_p) >= p.positive_c_max
                or c_e.min() <= 0):
            return time[:k], voltage[:k]

        ce_average = electrolyte.C_average @ c_e
        ce_sqrt_average = electrolyte.C_average @ np.sqrt(c_e)
        voltage[k] = cell_voltage(I, c_ss_n, c_ss_p, ce_average, ce_sqrt_average, p)["V"]
        if voltage[k] <= p.V_min or voltage[k] >= p.V_max:
            return time[:k+1], voltage[:k+1]

        # Hold this current until the next sample.
        if k < time.size - 1:
            c_s_n = negative.A @ c_s_n + negative.B[:, 0]*I
            c_s_p = positive.A @ c_s_p + positive.B[:, 0]*I
            c_e = electrolyte.A @ c_e + electrolyte.B[:, 0]*I

    return time, voltage


if __name__ == "__main__":
    # Example: 1C constant-current discharge with a 0.5 s time step.
    time = np.arange(0.0, 3600.5, 0.5)
    current = np.full(time.size, parameters.nominal_capacity_Ah)
    time, Vt = main_vt(time, current)
    print(f"{time[-1]:g} s: {Vt[-1]:.4f} V")
