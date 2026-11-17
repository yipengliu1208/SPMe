"""Build radial particle FDM matrices and internal node coordinates.
Convention: each column is one time sample; states are absolute concentrations.
Positive total current means discharge.

Variable reference:
    sign: +1 negative, -1 positive.
    epsilon_s: active solid volume fraction [-].
    R_s: particle radius [m]; D_s: solid diffusivity [m^2/s].
    L: electrode thickness [m].
    Area: electrode cross-sectional area [m^2]; F: Faraday constant [C/mol].
    n: number of internal radial states; Nt: number of time samples.
    dr: radial spacing Delta r=R_s/(n+1) [m].
    q: one-based integer node index during matrix assembly [-].
    A: discrete state matrix, shape (n,n), dimensionless.
    B: discrete current-input matrix, shape (n,1), [mol/(m^3 A)].
    C: surface state-output matrix, shape (1,n), dimensionless.
    D: surface current feedthrough matrix, shape (1,1), [mol/(m^3 A)].
    r: internal node radii q*dr [m], shape (n,).
    SolidModel / model: object containing matrices and radial coordinates.

    dt: state propagation interval [s], nonnegative; zero gives the identity update.
    M: augmented continuous matrix [[A_c,B_c],[0,0]].
    transition: state transition matrix containing discrete A and B; C and D are unchanged.
    A_c / B_c: assembled continuous matrices [1/s] and [mol/(m^3 A s)].
    Returned A / B: discrete matrices, dimensionless and [mol/(m^3 A)].

Equations: X_next=A@X+B*I; c_ss=C@X+D*I.
Continuous reference: dX/dt=A_c@X+B_c*I.
The @ operator is matrix multiplication; * is scalar or elementwise multiplication.

State nodes lie at r_q=q*dr, q=1,...,n; the last state is inside the particle.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm


@dataclass
class SolidModel:
    A: np.ndarray
    B: np.ndarray
    A_c: np.ndarray
    B_c: np.ndarray
    C: np.ndarray
    D: np.ndarray
    r: np.ndarray

# FDM
def build_solid_model(R_s, D_s, epsilon_s, L, sign, Area, n=60, *, dt, F):
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("The ZOH interval must be finite and nonnegative.")
    if int(n) != n or n < 3:
        raise ValueError("The derived surface stencil requires n >= 3.")
    n = int(n)
    dr = R_s / (n + 1)

    ## A matrix
    A_c = np.zeros((n, n))
    for q in range(1, n):
        A_c[q-1, q-1] = -2 * D_s / dr**2
        A_c[q-1, q] = (q+1)/q * (D_s / dr**2)
        if q > 1:
            A_c[q-1, q-2] = (q-1)/q * (D_s / dr**2)

    # Last row
    A_c[-1, -2] = 2*(n-2)/(3*n) * (D_s / dr**2)
    A_c[-1, -1] = -2*(n-2)/(3*n) * (D_s / dr**2)

    ## B matrix
    B_c = np.zeros((n, 1))
    B_c[-1, 0] = (-sign * 2*(n+1)*R_s / (9*n*dr * (F*epsilon_s*Area*L)))

    # C, D: c_ss = (4*c_n-c_(n-1))/3 - 2*dr*j_molar/(3*D_s).
    C = np.zeros((1, n))
    C[0, -2:] = [-1/3, 4/3]
    D = np.array([[-sign * 2*R_s*dr / (9*D_s * (F*epsilon_s*Area*L))]])


    # From continuous to discretized
    M = np.zeros((A_c.shape[0]+1, A_c.shape[0]+1))
    M[:-1, :-1] = A_c
    M[:-1, -1:] = B_c
    transition = expm(M*dt)

    return SolidModel(
        A=transition[:-1, :-1],
        B=transition[:-1, -1:],
        C=C,
        D=D,
        r=np.arange(1, n+1, dtype=float)*dr,
        A_c=A_c,
        B_c=B_c,
    )
