"""Build conservative FVM matrices for the three electrolyte regions.
States are absolute cell-average concentrations in negative/separator/positive order.

Variable reference:
    cell_parameters: cell-parameter module; F: Faraday constant [C/mol].
    counts: regional cell-count tuple; Nn/Ns/Np: negative/separator/positive cell counts.
    N: total number of electrolyte cells Nn+Ns+Np.
    n: current cell count during input validation.
    Ln / Ls / Lp: regional thicknesses [m].
    en / es / ep: regional electrolyte porosities [-].
    Dn / Ds / Dp: regional effective electrolyte diffusivities [m^2/s].
        Here Ds refers to separator electrolyte diffusion, not particle solid D_s.
    hn / hs / hp: regional cell widths [m].
    A_n / A_sep / A_p: regional state blocks, shapes
        (Nn,Nn)/(Ns,Ns)/(Np,Np), [1/s].
    A_ns / A_sn: interface coupling blocks, shapes (Nn,Ns)/(Ns,Nn), [1/s].
    A_sp / A_ps: interface coupling blocks, shapes (Ns,Np)/(Np,Ns), [1/s].
    A: complete discrete state matrix, shape (N,N), dimensionless.
    B_n / B_sep / B_p: regional current-input blocks, shapes
        (Nn,1)/(Ns,1)/(Np,1), [mol/(m^3 A s)].
    B: complete discrete current-input matrix, shape (N,1), [mol/(m^3 A)].
    C: six boundary-concentration outputs, shape (6,N), dimensionless.
        Both sides of each interface are included.
    D: boundary-output feedthrough matrix, shape (6,1), [mol/(m^3 A)], all zero.
    C_average: regional thickness-average matrix, shape (3,N), dimensionless.
    dx: electrolyte cell widths [m], shape (N,).
    centers: cell-center positions from the negative collector [m], shape (N,).
    blocks: dictionary storing regional and coupling matrices under their names.
    ElectrolyteModel: object containing matrices, blocks and mesh information.

    dt: state propagation interval [s], nonnegative; zero gives the identity update.
    M: augmented continuous matrix [[A_c,B_c],[0,0]].
    transition: state transition matrix containing discrete A and B; C and D are unchanged.
    A_c / B_c: assembled continuous matrices [1/s] and [mol/(m^3 A s)].
    Returned A / B: discrete matrices, dimensionless and [mol/(m^3 A)].

Equations: x_e_next=A@x_e+B*I; boundary outputs=C@x_e+D*I.
Continuous reference: dx_e/dt=A_c@x_e+B_c*I.
x_e has shape (N,Nt) and units [mol/m^3]; I is total current [A];
Nt is the number of time samples.
C_average@sqrt(x_e) computes the mean square root.
Mean(sqrt(c_e)) generally differs from sqrt(mean(c_e)).
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm


@dataclass
class ElectrolyteModel:
    A: np.ndarray
    B: np.ndarray
    A_c: np.ndarray
    B_c: np.ndarray
    C: np.ndarray
    D: np.ndarray
    C_average: np.ndarray
    blocks: dict
    dx: np.ndarray
    centers: np.ndarray
    counts: tuple


def build_electrolyte_model(cell_parameters, counts=(30, 10, 30), *, dt):
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("The ZOH interval must be finite and nonnegative.")
    # Checking
    if len(counts) != 3 or any(int(n) != n or n < 2 for n in counts):
        raise ValueError("Use at least two FVM cells in EACH region.")
    
    # Number of control volumes/segment in the three regions
    Nn = int(counts[0])
    Ns = int(counts[1])
    Np = int(counts[2])
    counts = (Nn, Ns, Np)

    # Thickness of each region
    Ln = cell_parameters.L_e[0]
    Ls = cell_parameters.L_e[1]
    Lp = cell_parameters.L_e[2]
    
    # Porosity, epsilon
    en = cell_parameters.epsilon_e[0]
    es = cell_parameters.epsilon_e[1]
    ep = cell_parameters.epsilon_e[2]

    # Effective diffusivities
    Dn = cell_parameters.D_e_eff[0]
    Ds = cell_parameters.D_e_eff[1]
    Dp = cell_parameters.D_e_eff[2]
    
    # Thickness of each control volume/segment
    hn = Ln / Nn
    hs = Ls / Ns
    hp = Lp / Np

    ## Negative electrode
    A_n = Dn / (en * hn**2) * (np.diag(-2*np.ones(Nn)) + np.diag(np.ones(Nn-1), 1) + np.diag(np.ones(Nn-1), -1))
    
    # collector boundary and separator interface
    A_n[0, 0] = -Dn / (en * hn**2)
    A_n[-1, -1] = (-Dn / (en * hn**2)-2 / (en * hn * (hn/Dn + hs/Ds)))

    ## Separator
    A_sep = Ds / (es * hs**2) * (np.diag(-2*np.ones(Ns)) + np.diag(np.ones(Ns-1), 1) + np.diag(np.ones(Ns-1), -1))
    
    # At both electrode interfaces
    A_sep[0, 0] = (
        -Ds / (es * hs**2)
        -2 / (es * hs * (hn/Dn + hs/Ds))
    )
    A_sep[-1, -1] = (
        -Ds / (es * hs**2)
        -2 / (es * hs * (hs/Ds + hp/Dp))
    )

    ## Positive electrode: separator interface and collector zero-flux boundary.
    A_p = Dp / (ep * hp**2) * (np.diag(-2*np.ones(Np)) + np.diag(np.ones(Np-1), 1) + np.diag(np.ones(Np-1), -1))
    
    # separator interface and collector boundary.
    A_p[0, 0] = (
        -Dp / (ep * hp**2)
        -2 / (ep * hp * (hs/Ds + hp/Dp))
    )
    A_p[-1, -1] = -Dp / (ep * hp**2)

    # Coupling blocks for the two interfaces
    A_ns = np.zeros((Nn, Ns))
    A_ns[-1, 0] = 2 / (en * hn * (hn/Dn + hs/Ds))
    A_sn = np.zeros((Ns, Nn))
    A_sn[0, -1] = 2 / (es * hs * (hn/Dn + hs/Ds))
    A_sp = np.zeros((Ns, Np))
    A_sp[-1, 0] = 2 / (es * hs * (hs/Ds + hp/Dp))
    A_ps = np.zeros((Np, Ns))
    A_ps[0, -1] = 2 / (ep * hp * (hs/Ds + hp/Dp))
    
    # Final A matrix
    A_c = np.block([[A_n, A_ns, np.zeros((Nn, Np))],
                  [A_sn, A_sep, A_sp],
                  [np.zeros((Np, Nn)), A_ps, A_p]])

    ## B matrix
    # Separator has no source term
    B_n = np.full((Nn, 1), (1-cell_parameters.t_plus) / (cell_parameters.F*cell_parameters.Area*en*Ln))
    B_sep = np.zeros((Ns, 1))
    B_p = np.full((Np, 1), -(1-cell_parameters.t_plus) / (cell_parameters.F*cell_parameters.Area*ep*Lp))
    B_c = np.vstack([B_n, B_sep, B_p])
    
    
    N = Nn + Ns + Np
    ## C matrix
    C = np.zeros((6, N))
    C[0, 0] = 1
    C[5, -1] = 1
    # Negative electrode-separator interface
    C[1, Nn-1] = (hs/Ds) / (hn/Dn + hs/Ds)
    C[1, Nn] = (hn/Dn) / (hn/Dn + hs/Ds)
    C[2] = C[1]
    
    # Positive electrode-separator interface
    C[3, Nn+Ns-1] = (hp/Dp) / (hs/Ds + hp/Dp)
    C[3, Nn+Ns] = (hs/Ds) / (hs/Ds + hp/Dp)
    C[4] = C[3]
    
    ## D matrix
    D = np.zeros((6, 1))


    # mean c_e is for electrolyte potential
    # mean sqrt c_e is for exchange current density
    C_average = np.zeros((3, N))
    C_average[0, :Nn] = hn/Ln
    C_average[1, Nn:Nn+Ns] = hs/Ls
    C_average[2, Nn+Ns:] = hp/Lp
    dx = np.repeat([hn, hs, hp], counts)
    centers = np.cumsum(dx) - dx/2
    blocks = {
        "A_n": A_n, 
        "A_sep": A_sep,
        "A_p": A_p,
        "A_ns": A_ns,
        "A_sn": A_sn,
        "A_sp": A_sp,
        "A_ps": A_ps,
        "B_n": B_n,
        "B_sep": B_sep,
        "B_p": B_p,
    }
    
    # From continuous to discretized
    M = np.zeros((A_c.shape[0]+1, A_c.shape[0]+1))
    M[:-1, :-1] = A_c
    M[:-1, -1:] = B_c
    transition = expm(M*dt)

    return ElectrolyteModel(
        A=transition[:-1, :-1], # N x N
        B=transition[:-1, -1:], # N x 1
        C=C, # 6 x N
        D=D, # 6 x 1
        C_average=C_average,
        blocks=blocks,
        dx=dx,
        centers=centers,
        counts=counts,
        A_c=A_c,
        B_c=B_c,
    )
