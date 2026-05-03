import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numba import njit


# ============================================================
# USER PARAMETERS
# ============================================================

# "Approximation by 8 harmonics" in the paper corresponds to 2N = 8,
# hence N = 4.
N = 4

RT = 9110
RS = 8000.0
SIGMA = 1.0
KAPPA = 1.0 / np.sqrt(10.0)

# Time marching.
# Section VI of Sibgatullin et al. uses the asymptotic window t = 100..300.
# DT=2e-4 is conservative. If the run is too slow, first try DT=5e-4.
DT = 2.0e-4
T_TOTAL = 300.0
SAVE_EVERY = 50

# Plotting windows.
PHASE_TMIN = 100.0
NUSS_TMIN = 100.0
NUSS_TMAX = 300.0
# Optional custom y-limits for Nusselt plot. Use None for automatic.
# Example: NUSS_YLIM = (1.0, 3.2)
NUSS_YLIM = None
# Optional custom 3D axis limits. Use None for automatic.
PHASE3D_TAU_LIM = None
PHASE3D_PSI_LIM = None
PHASE3D_NUT_LIM = None
PHASE3D_NUS_LIM = None
PHASE3D_SAL_LIM = None

# Poincare section.
TAU_SECTION = -0.36
SECTION_DIRECTION = 0  # +1 upward, -1 downward, 0 both directions

# Multiple nearby trajectories for a thicker phase portrait.
# The paper uses one trajectory starting near the coordinate origin.
N_TRAJ = 4
PERT_MAIN = 0.0
PERT_SMALL = 0.0

# Longer reference trajectory for Nusselt/Poincare diagnostics.
REF_T_TOTAL = 300.0
REF_SAVE_EVERY = 50

# Important for the Poincare map: detect section crossings at every RK step,
# not only from downsampled saved output.
USE_STEP_LEVEL_POINCARE_EVENTS = True

# Quadrature grid used once to build coefficient operators.
NX = 64
NZ = 64

# Cache.
CACHE_DIR = Path("galerkin_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Important: new cache name so old wrong-basis tensor file is not reused.
# Different cache name because the paper-style Nusselt weights are different.
CACHE_FILE = CACHE_DIR / f"ops_N{N}_NX{NX}_NZ{NZ}_sqrt2basis_paper_flux_2j.npz"

# Set True for first run after changing basis/wavenumber.
# After the corrected cache is created, you may set this to False.
# Set True only if you changed N, NX, NZ, basis, or operator construction.
# For normal reruns after the cache exists, False saves time.
FORCE_REBUILD = False

# Output files.
OUT_PHASE = "corrected_coeff_phase.png"
OUT_NU = "corrected_coeff_nusselts.png"
OUT_POINCARE_REF = "corrected_coeff_poincare_ref.png"
OUT_POINCARE_REF_UP = "corrected_coeff_poincare_ref_upward.png"
OUT_POINCARE_REF_DOWN = "corrected_coeff_poincare_ref_downward.png"
OUT_POINCARE_EVENTS_CSV = "corrected_coeff_poincare_events_tau_m0p36.csv"
OUT_PHASE_3D_SAL = "corrected_coeff_phase_3d_sal.png"
OUT_PHASE_3D_NUT = "corrected_coeff_phase_3d_NuT.png"

# Interactive 3D HTML output.
# This gives true mouse rotation/zoom/pan in a browser.
MAKE_INTERACTIVE_3D_HTML = False
OUT_PHASE_3D_SAL_HTML = "corrected_coeff_phase_3d_sal_interactive.html"
OUT_PHASE_3D_NUT_HTML = "corrected_coeff_phase_3d_NuT_interactive.html"


# More dramatic nebula-style cosmic visualization + optional video export.
MAKE_COSMIC_NEBULA_HTML = False
# HTML is fast. GIF/MP4 export is slower because it renders many PNG frames.
MAKE_COSMIC_GIF = False
MAKE_COSMIC_MP4 = False

OUT_COSMIC_NEBULA_HTML = "corrected_coeff_phase_3d_nebula_cosmic.html"
OUT_COSMIC_GIF = "corrected_coeff_phase_3d_nebula_cosmic.gif"
OUT_COSMIC_MP4 = "corrected_coeff_phase_3d_nebula_cosmic.mp4"


# Extra artistic view in the spirit of common strange-attractor demos.
# The old 2D-only still image is intentionally not produced anymore.
MAKE_ARTISTIC_STYLE_STILL = False
OUT_ARTISTIC_STYLE_STILL = "corrected_coeff_phase_artistic_style_still.png"
OUT_ARTISTIC_STYLE_3D_SAL = "corrected_coeff_phase_3d_sal_artistic_style.png"


# Animation with bright "electrons" moving along the projected phase trajectories.
# GIF is enabled by default; MP4 is optional.
MAKE_ARTISTIC_STYLE_ELECTRONS_GIF = False
MAKE_ARTISTIC_STYLE_ELECTRONS_MP4 = False

OUT_ARTISTIC_STYLE_ELECTRONS_GIF = "corrected_coeff_phase_3d_sal_electrons.gif"
OUT_ARTISTIC_STYLE_ELECTRONS_MP4 = "corrected_coeff_phase_3d_sal_electrons.mp4"


# ============================================================
# MODE LISTS
# ============================================================

def build_psi_modes(N):
    """
    Streamfunction modes:
        i >= 1, j >= 1
        i + j even
        i^2 + j^2 <= (2N)^2
    """
    modes = []
    lim = 2 * N

    for i in range(1, lim + 1):
        for j in range(1, lim + 1):
            if ((i + j) % 2 == 0) and (i * i + j * j <= lim * lim):
                modes.append((i, j))

    return modes


def build_scalar_modes(N):
    """
    Temperature/salinity modes:
        i >= 0, j >= 1
        i + j even
        i^2 + j^2 <= (2N)^2
    """
    modes = []
    lim = 2 * N

    for i in range(0, lim + 1):
        for j in range(1, lim + 1):
            if ((i + j) % 2 == 0) and (i * i + j * j <= lim * lim):
                modes.append((i, j))

    return modes


PSI_MODES = build_psi_modes(N)
SC_MODES = build_scalar_modes(N)

MPSI = len(PSI_MODES)
MSC = len(SC_MODES)
DIM = MPSI + 2 * MSC

psi11_idx = PSI_MODES.index((1, 1))
tau11_idx = SC_MODES.index((1, 1))
sal11_idx = SC_MODES.index((1, 1))

print(f"N = {N}")
print(f"2N cutoff    = {2*N}")
print(f"psi modes    = {MPSI}")
print(f"scalar modes = {MSC}")
print(f"total dim    = {DIM}")
print(f"psi modes list = {PSI_MODES}")
print(f"scalar modes list = {SC_MODES}")


# ============================================================
# BASIS ARRAYS FOR ONE-TIME PRECOMPUTE
# ============================================================

def build_basis_arrays():
    """
    Build basis and derivative arrays on the quadrature grid.

    Correct paper domain:
        x ∈ [0, 2√2)
        z ∈ [0, 1]

    Correct horizontal wavenumber:
        ax = iπ/√2
    """

    # Correct period for sin(iπx/√2), cos(iπx/√2).
    Lx = 2.0 * np.sqrt(2.0)

    x = np.linspace(0.0, Lx, NX, endpoint=False)
    z = np.linspace(0.0, 1.0, NZ + 1)

    dx = x[1] - x[0]
    dz = z[1] - z[0]

    X, Z = np.meshgrid(x, z)

    # Trapezoidal weights in z, periodic rectangle rule in x.
    wz = np.ones(NZ + 1, dtype=np.float64)
    wz[0] = 0.5
    wz[-1] = 0.5

    W = (wz[:, None] * dx * dz).astype(np.float64)

    # Streamfunction basis.
    psi_basis = np.zeros((MPSI, NZ + 1, NX), dtype=np.float64)
    psi_x_basis = np.zeros_like(psi_basis)
    psi_z_basis = np.zeros_like(psi_basis)
    lap_psi_x_basis = np.zeros_like(psi_basis)
    lap_psi_z_basis = np.zeros_like(psi_basis)
    lam_psi = np.zeros(MPSI, dtype=np.float64)

    for m, (i, j) in enumerate(PSI_MODES):
        ax = i * np.pi / np.sqrt(2.0)
        bz = j * np.pi

        # Laplacian eigenvalue:
        # Δ basis = lam * basis, lam < 0
        lam = -(ax * ax + bz * bz)

        sxi = np.sin(ax * X)
        cxi = np.cos(ax * X)

        szj = np.sin(bz * Z)
        czj = np.cos(bz * Z)

        psi_basis[m] = sxi * szj
        psi_x_basis[m] = ax * cxi * szj
        psi_z_basis[m] = bz * sxi * czj

        lap_psi_x_basis[m] = lam * psi_x_basis[m]
        lap_psi_z_basis[m] = lam * psi_z_basis[m]

        lam_psi[m] = lam

    # Scalar basis.
    sc_basis = np.zeros((MSC, NZ + 1, NX), dtype=np.float64)
    sc_x_basis = np.zeros_like(sc_basis)
    sc_z_basis = np.zeros_like(sc_basis)
    lam_sc = np.zeros(MSC, dtype=np.float64)

    for m, (i, j) in enumerate(SC_MODES):
        ax = i * np.pi / np.sqrt(2.0)
        bz = j * np.pi

        lam = -(ax * ax + bz * bz)

        cxi = np.cos(ax * X)
        sxi = np.sin(ax * X)

        szj = np.sin(bz * Z)
        czj = np.cos(bz * Z)

        sc_basis[m] = cxi * szj
        sc_x_basis[m] = -ax * sxi * szj
        sc_z_basis[m] = bz * cxi * czj

        lam_sc[m] = lam

    psi_norm = np.einsum("zx,mzx,mzx->m", W, psi_basis, psi_basis, optimize=True)
    sc_norm = np.einsum("zx,mzx,mzx->m", W, sc_basis, sc_basis, optimize=True)

    return {
        "W": W,
        "psi_basis": psi_basis,
        "psi_x_basis": psi_x_basis,
        "psi_z_basis": psi_z_basis,
        "lap_psi_x_basis": lap_psi_x_basis,
        "lap_psi_z_basis": lap_psi_z_basis,
        "lam_psi": lam_psi,
        "sc_basis": sc_basis,
        "sc_x_basis": sc_x_basis,
        "sc_z_basis": sc_z_basis,
        "lam_sc": lam_sc,
        "psi_norm": psi_norm,
        "sc_norm": sc_norm,
    }


# ============================================================
# ONE-TIME OPERATOR BUILD
# ============================================================

def build_operators():
    """
    Build coefficient-space tensors.

    Let:
        a = psi coefficients
        b = tau coefficients
        c = salinity coefficients

    Coefficient ODEs:

        a_dot =
            σ λψ a
            - σ R_T * projection(tau_x)/λψ
            + σ R_S * projection(s_x)/λψ
            + projection(J(ψ, Δψ))/λψ

        b_dot =
            λτ b - projection(ψ_x) + projection(J(ψ, τ))

        c_dot =
            k λs c - projection(ψ_x) + projection(J(ψ, s))

    where λψ, λτ, λs are negative Laplacian eigenvalues.
    """

    t0 = time.time()
    arr = build_basis_arrays()
    W = arr["W"]

    psi_basis = arr["psi_basis"]
    psi_x_basis = arr["psi_x_basis"]
    psi_z_basis = arr["psi_z_basis"]
    lap_psi_x_basis = arr["lap_psi_x_basis"]
    lap_psi_z_basis = arr["lap_psi_z_basis"]
    lam_psi = arr["lam_psi"]

    sc_basis = arr["sc_basis"]
    sc_x_basis = arr["sc_x_basis"]
    sc_z_basis = arr["sc_z_basis"]
    lam_sc = arr["lam_sc"]

    psi_norm = arr["psi_norm"]
    sc_norm = arr["sc_norm"]

    print("Building corrected coefficient-space operators...")

    # Quadratic tensor for psi equation:
    # projection of J(psi, Δpsi), divided by Laplacian eigenvalue.
    part1 = np.einsum(
        "zx,mzx,izx,jzx->mij",
        W,
        psi_basis,
        psi_x_basis,
        lap_psi_z_basis,
        optimize=True,
    )

    part2 = np.einsum(
        "zx,mzx,izx,jzx->mij",
        W,
        psi_basis,
        psi_z_basis,
        lap_psi_x_basis,
        optimize=True,
    )

    Qpsi = (part1 - part2) / psi_norm[:, None, None]
    Qpsi = Qpsi / lam_psi[:, None, None]

    # Quadratic tensor for tau equation:
    # projection of J(psi, tau).
    part1 = np.einsum(
        "zx,mzx,izx,jzx->mij",
        W,
        sc_basis,
        psi_x_basis,
        sc_z_basis,
        optimize=True,
    )

    part2 = np.einsum(
        "zx,mzx,izx,jzx->mij",
        W,
        sc_basis,
        psi_z_basis,
        sc_x_basis,
        optimize=True,
    )

    Qtau = (part1 - part2) / sc_norm[:, None, None]
    Qsal = Qtau.copy()

    # Linear coupling from tau_x / s_x into psi equation,
    # already divided by Laplacian eigenvalue.
    Bpsi_sc = np.einsum(
        "zx,mzx,jzx->mj",
        W,
        psi_basis,
        sc_x_basis,
        optimize=True,
    )

    Bpsi_sc = Bpsi_sc / psi_norm[:, None]
    Bpsi_sc = Bpsi_sc / lam_psi[:, None]

    # Linear forcing from -psi_x into scalar equations.
    Fsc_psi = -np.einsum(
        "zx,mzx,izx->mi",
        W,
        sc_basis,
        psi_x_basis,
        optimize=True,
    )

    Fsc_psi = Fsc_psi / sc_norm[:, None]

    # Nusselt extraction weights.
    # Section VI writes the horizontally averaged boundary fluxes as
    #     Nu = 1 - 2 * sum_j j * tau_0j,
    #     Ns = 1 - 2 * sum_j j * s_0j.
    # Only i=0 scalar modes contribute after averaging over x.
    # If you want the literal derivative of sin(j*pi*z), change 2.0*j to j*np.pi.
    nu_weights = np.array(
        [2.0 * j if i == 0 else 0.0 for (i, j) in SC_MODES],
        dtype=np.float64,
    )

    print(f"Operator build done in {time.time() - t0:.2f} s")

    np.savez_compressed(
        CACHE_FILE,
        lam_psi=lam_psi,
        lam_sc=lam_sc,
        Qpsi=Qpsi,
        Qtau=Qtau,
        Qsal=Qsal,
        Bpsi_sc=Bpsi_sc,
        Fsc_psi=Fsc_psi,
        nu_weights=nu_weights,
    )

    print(f"Saved corrected operator cache: {CACHE_FILE}")


def load_operators():
    if FORCE_REBUILD or (not CACHE_FILE.exists()):
        build_operators()

    data = np.load(CACHE_FILE)
    ops = {key: data[key] for key in data.files}

    print(f"Loaded operator cache: {CACHE_FILE}")
    return ops


# ============================================================
# FAST COEFFICIENT-SPACE RHS
# ============================================================

@njit
def rhs_coeff(
    y,
    sigma,
    RT,
    RS,
    kappa,
    lam_psi,
    lam_sc,
    Qpsi,
    Qtau,
    Qsal,
    Bpsi_sc,
    Fsc_psi,
):
    a = y[:MPSI]
    b = y[MPSI:MPSI + MSC]
    c = y[MPSI + MSC:]

    da = np.zeros(MPSI, dtype=np.float64)
    db = np.zeros(MSC, dtype=np.float64)
    dc = np.zeros(MSC, dtype=np.float64)

    # Streamfunction equation.
    for m in range(MPSI):
        val = sigma * lam_psi[m] * a[m]

        # Rayleigh coupling.
        # Full Galerkin system uses RT and RS directly.
        for j in range(MSC):
            val += (-sigma * RT) * Bpsi_sc[m, j] * b[j]
            val += (+sigma * RS) * Bpsi_sc[m, j] * c[j]

        # Nonlinear J(psi, Δpsi).
        q = 0.0
        for i in range(MPSI):
            ai = a[i]
            for j in range(MPSI):
                q += Qpsi[m, i, j] * ai * a[j]

        da[m] = val + q

    # Temperature equation.
    for m in range(MSC):
        val = lam_sc[m] * b[m]

        # -psi_x
        for i in range(MPSI):
            val += Fsc_psi[m, i] * a[i]

        # J(psi, tau)
        q = 0.0
        for i in range(MPSI):
            ai = a[i]
            for j in range(MSC):
                q += Qtau[m, i, j] * ai * b[j]

        db[m] = val + q

    # Salinity equation.
    for m in range(MSC):
        val = kappa * lam_sc[m] * c[m]

        # -psi_x
        for i in range(MPSI):
            val += Fsc_psi[m, i] * a[i]

        # J(psi, s)
        q = 0.0
        for i in range(MPSI):
            ai = a[i]
            for j in range(MSC):
                q += Qsal[m, i, j] * ai * c[j]

        dc[m] = val + q

    out = np.empty(DIM, dtype=np.float64)
    out[:MPSI] = da
    out[MPSI:MPSI + MSC] = db
    out[MPSI + MSC:] = dc

    return out


@njit
def rk4_step_coeff(
    y,
    dt,
    sigma,
    RT,
    RS,
    kappa,
    lam_psi,
    lam_sc,
    Qpsi,
    Qtau,
    Qsal,
    Bpsi_sc,
    Fsc_psi,
):
    k1 = rhs_coeff(
        y,
        sigma,
        RT,
        RS,
        kappa,
        lam_psi,
        lam_sc,
        Qpsi,
        Qtau,
        Qsal,
        Bpsi_sc,
        Fsc_psi,
    )

    k2 = rhs_coeff(
        y + 0.5 * dt * k1,
        sigma,
        RT,
        RS,
        kappa,
        lam_psi,
        lam_sc,
        Qpsi,
        Qtau,
        Qsal,
        Bpsi_sc,
        Fsc_psi,
    )

    k3 = rhs_coeff(
        y + 0.5 * dt * k2,
        sigma,
        RT,
        RS,
        kappa,
        lam_psi,
        lam_sc,
        Qpsi,
        Qtau,
        Qsal,
        Bpsi_sc,
        Fsc_psi,
    )

    k4 = rhs_coeff(
        y + dt * k3,
        sigma,
        RT,
        RS,
        kappa,
        lam_psi,
        lam_sc,
        Qpsi,
        Qtau,
        Qsal,
        Bpsi_sc,
        Fsc_psi,
    )

    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# OBSERVABLES
# ============================================================

def observables_fast(y, nu_weights):
    """
    Extract low-mode observables used for phase portraits and diagnostics.

    psi11 : streamfunction coefficient psi_11
    tau11 : temperature coefficient tau_11
    sal11 : salinity coefficient s_11
    NuT   : thermal Nusselt number proxy
    NuS   : salinity Nusselt number proxy
    """

    a = y[:MPSI]
    b = y[MPSI:MPSI + MSC]
    c = y[MPSI + MSC:]

    psi11 = a[psi11_idx]
    tau11 = b[tau11_idx]
    sal11 = c[sal11_idx]

    NuT = 1.0 - np.dot(nu_weights, b)
    NuS = 1.0 - np.dot(nu_weights, c)

    return psi11, tau11, sal11, NuT, NuS


# ============================================================
# INITIAL CONDITIONS
# ============================================================

def build_base_ic():
    """
    Exact initial condition stated in Section VI of Sibgatullin et al. (2003):
        psi_11(0) = 1e-6,
        all other psi_ij(0) = 0,
        all tau_ij(0) = 0,
        all s_ij(0) = 0.
    """
    y0 = np.zeros(DIM, dtype=np.float64)
    y0[psi11_idx] = 1.0e-6
    return y0


def build_traj_ic(base_y0, seed_shift):
    rng = np.random.default_rng(1000 + seed_shift)
    y = base_y0.copy()

    if N_TRAJ <= 1:
        return y

    # Structured perturbation of dominant modes.
    y[psi11_idx] += PERT_MAIN * rng.standard_normal()
    y[MPSI + tau11_idx] += PERT_MAIN * rng.standard_normal()
    y[MPSI + MSC + sal11_idx] += PERT_MAIN * rng.standard_normal()

    # Tiny perturbation of first few low modes.
    npsi = min(4, MPSI)
    nsc = min(5, MSC)

    y[:npsi] += PERT_SMALL * rng.standard_normal(npsi)
    y[MPSI:MPSI + nsc] += PERT_SMALL * rng.standard_normal(nsc)
    y[MPSI + MSC:MPSI + MSC + nsc] += PERT_SMALL * rng.standard_normal(nsc)

    return y


# ============================================================
# TRAJECTORY INTEGRATION
# ============================================================

def integrate_one_trajectory(y0, ops, traj_id=0):
    nsteps = int(round(T_TOTAL / DT))

    psi_hist = []
    tau_hist = []
    sal_hist = []
    sal_hist = []
    NuT_hist = []
    NuS_hist = []
    t_hist = []

    y = y0.copy()
    blown = False

    t0 = time.time()
    print(f"Integrating trajectory {traj_id + 1}/{N_TRAJ}...")

    for n in range(nsteps):
        y = rk4_step_coeff(
            y,
            DT,
            SIGMA,
            RT,
            RS,
            KAPPA,
            ops["lam_psi"],
            ops["lam_sc"],
            ops["Qpsi"],
            ops["Qtau"],
            ops["Qsal"],
            ops["Bpsi_sc"],
            ops["Fsc_psi"],
        )

        if not np.all(np.isfinite(y)):
            blown = True
            print(f"NaN/Inf detected in trajectory {traj_id+1} at step {n}")
            break

        ymax = np.max(np.abs(y))
        if ymax > 1e8:
            blown = True
            print(f"Blow-up in trajectory {traj_id+1} at step {n}, max|y|={ymax:.3e}")
            break

        if n % SAVE_EVERY == 0:
            psi11, tau11, sal11, NuT, NuS = observables_fast(y, ops["nu_weights"])
            psi_hist.append(psi11)
            tau_hist.append(tau11)
            sal_hist.append(sal11)
            NuT_hist.append(NuT)
            NuS_hist.append(NuS)
            t_hist.append(n * DT)

    print(f"  done in {time.time() - t0:.2f} s")

    return {
        "blown": blown,
        "t": np.asarray(t_hist),
        "psi": np.asarray(psi_hist),
        "tau": np.asarray(tau_hist),
        "sal": np.asarray(sal_hist),
        "NuT": np.asarray(NuT_hist),
        "NuS": np.asarray(NuS_hist),
    }


def integrate_reference_trajectory(y0, ops):
    nsteps = int(round(REF_T_TOTAL / DT))

    psi_hist = []
    tau_hist = []
    sal_hist = []
    NuT_hist = []
    NuS_hist = []
    t_hist = []

    y = y0.copy()
    blown = False

    print("Integrating long reference trajectory for diagnostics...")

    for n in range(nsteps):
        y = rk4_step_coeff(
            y,
            DT,
            SIGMA,
            RT,
            RS,
            KAPPA,
            ops["lam_psi"],
            ops["lam_sc"],
            ops["Qpsi"],
            ops["Qtau"],
            ops["Qsal"],
            ops["Bpsi_sc"],
            ops["Fsc_psi"],
        )

        if not np.all(np.isfinite(y)):
            blown = True
            print(f"NaN/Inf detected in reference trajectory at step {n}")
            break

        ymax = np.max(np.abs(y))
        if ymax > 1e8:
            blown = True
            print(f"Blow-up in reference trajectory at step {n}, max|y|={ymax:.3e}")
            break

        if n % REF_SAVE_EVERY == 0:
            psi11, tau11, sal11, NuT, NuS = observables_fast(y, ops["nu_weights"])
            psi_hist.append(psi11)
            tau_hist.append(tau11)
            sal_hist.append(sal11)
            NuT_hist.append(NuT)
            NuS_hist.append(NuS)
            t_hist.append(n * DT)

    return {
        "blown": blown,
        "t": np.asarray(t_hist),
        "psi": np.asarray(psi_hist),
        "tau": np.asarray(tau_hist),
        "sal": np.asarray(sal_hist),
        "NuT": np.asarray(NuT_hist),
        "NuS": np.asarray(NuS_hist),
    }


# ============================================================
# POINCARE SECTION
# ============================================================

def section_points_tau_fixed(t, tau, psi, tau_section, direction=+1, t_min=0.0):
    psi_cross = []
    t_cross = []

    for i in range(len(t) - 1):
        if t[i] < t_min:
            continue

        tau0 = tau[i]
        tau1 = tau[i + 1]
        psi0 = psi[i]
        psi1 = psi[i + 1]

        if direction == +1:
            crossed = (tau0 < tau_section) and (tau1 >= tau_section)
        elif direction == -1:
            crossed = (tau0 > tau_section) and (tau1 <= tau_section)
        else:
            crossed = (
                ((tau0 < tau_section) and (tau1 >= tau_section))
                or ((tau0 > tau_section) and (tau1 <= tau_section))
            )

        if crossed:
            denom = tau1 - tau0
            if abs(denom) < 1e-14:
                continue

            alpha = (tau_section - tau0) / denom
            psii = psi0 + alpha * (psi1 - psi0)
            ti = t[i] + alpha * (t[i + 1] - t[i])

            psi_cross.append(psii)
            t_cross.append(ti)

    return np.asarray(t_cross), np.asarray(psi_cross)


def section_points_tau_fixed_observables(t, tau, psi, NuT, NuS, tau_section, direction=0, t_min=0.0):
    """
    Intersections with tau_11 = tau_section from one time-ordered trajectory.
    Returns interpolated time, psi_11, Nu_T, and Nu_S at each crossing.
    This is the construction used for the paper-style succession map and
    the section-based Nusselt plots.
    """
    tc, pc, ntc, nsc = [], [], [], []
    for i in range(len(t) - 1):
        if t[i] < t_min:
            continue
        tau0, tau1 = tau[i], tau[i + 1]
        if direction == +1:
            crossed = (tau0 < tau_section) and (tau1 >= tau_section)
        elif direction == -1:
            crossed = (tau0 > tau_section) and (tau1 <= tau_section)
        else:
            crossed = ((tau0 < tau_section) and (tau1 >= tau_section)) or ((tau0 > tau_section) and (tau1 <= tau_section))
        if not crossed:
            continue
        denom = tau1 - tau0
        if abs(denom) < 1e-14:
            continue
        a = (tau_section - tau0) / denom
        tc.append(t[i] + a * (t[i + 1] - t[i]))
        pc.append(psi[i] + a * (psi[i + 1] - psi[i]))
        ntc.append(NuT[i] + a * (NuT[i + 1] - NuT[i]))
        nsc.append(NuS[i] + a * (NuS[i + 1] - NuS[i]))
    return np.asarray(tc), np.asarray(pc), np.asarray(ntc), np.asarray(nsc)



# ============================================================
# PLOTTING HELPERS
# ============================================================

def style_center_axes(ax, xticks=None, yticks=None):
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.spines["left"].set_color("0.55")
    ax.spines["bottom"].set_color("0.55")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.minorticks_off()

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=0.8,
        labelsize=11,
        colors="black",
        pad=4,
    )



def integrate_reference_with_step_events(y0, ops):
    """
    Integrate the reference trajectory and detect tau_11 = TAU_SECTION
    crossings at every RK step.
    """
    nsteps = int(round(REF_T_TOTAL / DT))

    psi_hist, tau_hist, sal_hist, NuT_hist, NuS_hist, t_hist = [], [], [], [], [], []
    ev_t, ev_psi, ev_nut, ev_nus, ev_dir = [], [], [], [], []

    y = y0.copy()
    psi0, tau0, sal0, NuT0, NuS0 = observables_fast(y, ops["nu_weights"])

    print("Integrating long reference trajectory with step-level Poincare events...")

    for n in range(nsteps):
        t0 = n * DT
        t1 = (n + 1) * DT

        y_new = rk4_step_coeff(
            y, DT, SIGMA, RT, RS, KAPPA,
            ops["lam_psi"], ops["lam_sc"],
            ops["Qpsi"], ops["Qtau"], ops["Qsal"],
            ops["Bpsi_sc"], ops["Fsc_psi"],
        )

        if not np.all(np.isfinite(y_new)):
            print(f"NaN/Inf detected in reference trajectory at step {n}")
            break

        psi1, tau1, sal1, NuT1, NuS1 = observables_fast(y_new, ops["nu_weights"])

        if t0 >= PHASE_TMIN:
            upward = (tau0 < TAU_SECTION) and (tau1 >= TAU_SECTION)
            downward = (tau0 > TAU_SECTION) and (tau1 <= TAU_SECTION)

            if upward or downward:
                denom = tau1 - tau0
                if abs(denom) > 1.0e-15:
                    a = (TAU_SECTION - tau0) / denom
                    if 0.0 <= a <= 1.0:
                        ev_t.append(t0 + a * DT)
                        ev_psi.append(psi0 + a * (psi1 - psi0))
                        ev_nut.append(NuT0 + a * (NuT1 - NuT0))
                        ev_nus.append(NuS0 + a * (NuS1 - NuS0))
                        ev_dir.append(+1 if upward else -1)

        if n % REF_SAVE_EVERY == 0:
            psi_hist.append(psi1)
            tau_hist.append(tau1)
            sal_hist.append(sal1)
            NuT_hist.append(NuT1)
            NuS_hist.append(NuS1)
            t_hist.append(t1)

        y = y_new
        psi0, tau0, sal0, NuT0, NuS0 = psi1, tau1, sal1, NuT1, NuS1

    events = {
        "t": np.asarray(ev_t),
        "psi": np.asarray(ev_psi),
        "NuT": np.asarray(ev_nut),
        "NuS": np.asarray(ev_nus),
        "direction": np.asarray(ev_dir, dtype=np.int64),
    }

    trajectory = {
        "blown": False,
        "t": np.asarray(t_hist),
        "psi": np.asarray(psi_hist),
        "tau": np.asarray(tau_hist),
        "sal": np.asarray(sal_hist),
        "NuT": np.asarray(NuT_hist),
        "NuS": np.asarray(NuS_hist),
    }

    return trajectory, events


def save_events_csv(events, outfile):
    data = np.column_stack([
        events["t"],
        events["psi"],
        events["NuT"],
        events["NuS"],
        events["direction"],
    ])
    header = "t,psi11_at_tau_section,NuT_at_tau_section,NuS_at_tau_section,direction(+1_up_-1_down)"
    np.savetxt(outfile, data, delimiter=",", header=header, comments="")
    print(f"Saved: {outfile}")


def plot_poincare_from_event_sequence(psi_cross, title, outfile):
    fig, ax = plt.subplots(figsize=(6.5, 6.0), facecolor="white")
    ax.set_facecolor("white")

    if len(psi_cross) >= 2:
        x = psi_cross[:-1]
        y = psi_cross[1:]
        ax.plot(x, y, "k.", ms=2.2)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        dx = max(xmax - xmin, 1.0e-4)
        dy = max(ymax - ymin, 1.0e-4)
        pad = 0.08 * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_normal_axes(ax)
    ax.set_xlabel(r"$\psi_{11}^{(n)}$", fontsize=18, labelpad=10)
    ax.set_ylabel(r"$\psi_{11}^{(n+1)}$", fontsize=18, labelpad=12)
    ax.set_title(title, fontsize=16, pad=12)
    fig.subplots_adjust(left=0.18, bottom=0.16, right=0.96, top=0.86)
    plt.savefig(outfile, dpi=300, facecolor="white", bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")



def style_normal_axes(ax, xticks=None, yticks=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.minorticks_off()

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=0.8,
        labelsize=11,
        colors="black",
        pad=4,
    )



# ============================================================
# 3D PHASE PORTRAIT
# ============================================================

def plot_phase_3d(
    trajectories,
    use="sal",
    tmin=PHASE_TMIN,
    outfile="corrected_coeff_phase_3d_sal.png",
    elev=24,
    azim=-58,
):
    """
    3D phase portrait for checking whether the 2D ribbon-like phase portrait
    is a projected twisted band.

    Coordinates:
        x = tau11
        y = psi11

    Third coordinate options:
        use="sal" -> z = s11
        use="NuT" -> z = NuT
        use="NuS" -> z = NuS

    For Möbius-band interpretation, the most natural first check is:
        (tau11, psi11, s11)

    If the apparent 2D crossing separates in 3D, the ribbon interpretation is
    much stronger than if all branches lie on top of one another.
    """

    fig = plt.figure(figsize=(9, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    zlabel = None
    title = None

    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue

        mask = tr["t"] >= tmin

        if not np.any(mask):
            continue

        x = tr["tau"][mask]
        y = tr["psi"][mask]

        if use == "sal":
            z = tr["sal"][mask]
            zlabel = "s11"
            title = r"3D phase portrait: $(\tau_{11}, \psi_{11}, s_{11})$"
        elif use == "NuT":
            z = tr["NuT"][mask]
            zlabel = r"$Nu_T$"
            title = r"3D phase portrait: $(\tau_{11}, \psi_{11}, Nu_T)$"
        elif use == "NuS":
            z = tr["NuS"][mask]
            zlabel = r"$Nu_S$"
            title = r"3D phase portrait: $(\tau_{11}, \psi_{11}, Nu_S)$"
        else:
            raise ValueError("use must be 'sal', 'NuT', or 'NuS'")

        ax.plot(x, y, z, color="black", lw=0.45, alpha=0.50)

    ax.set_xlabel(r"$\tau_{11}$", fontsize=13, labelpad=14)
    ax.set_ylabel(r"$\psi_{11}$", fontsize=13, labelpad=16)
    ax.set_zlabel(zlabel, fontsize=13, labelpad=18)

    # Optional custom axis limits. These are useful when comparing several cases.
    if PHASE3D_TAU_LIM is not None:
        ax.set_xlim(*PHASE3D_TAU_LIM)
    if PHASE3D_PSI_LIM is not None:
        ax.set_ylim(*PHASE3D_PSI_LIM)
    if use == "sal" and PHASE3D_SAL_LIM is not None:
        ax.set_zlim(*PHASE3D_SAL_LIM)
    if use == "NuT" and PHASE3D_NUT_LIM is not None:
        ax.set_zlim(*PHASE3D_NUT_LIM)
    if use == "NuS" and PHASE3D_NUS_LIM is not None:
        ax.set_zlim(*PHASE3D_NUS_LIM)

    ax.set_title(title, fontsize=18, pad=24)

    # Avoid visually stretched 3D boxes.
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)

    # Manual margins avoid clipped axis labels in mplot3d.
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.90)
    plt.savefig(outfile, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.show()

    print(f"Saved: {outfile}")


def plot_phase_3d_three_views(
    trajectories,
    use="sal",
    tmin=PHASE_TMIN,
    outfile_prefix="corrected_coeff_phase_3d_sal_view",
):
    """
    Save three different viewing angles of the same 3D phase portrait.

    This is useful because a Möbius-like strip may look ordinary from one
    direction but reveal the over/under separation from another direction.
    """

    views = [
        (24, -58),
        (18, 35),
        (55, -25),
    ]

    for idx, (elev, azim) in enumerate(views, start=1):
        outfile = f"{outfile_prefix}{idx}.png"
        plot_phase_3d(
            trajectories,
            use=use,
            tmin=tmin,
            outfile=outfile,
            elev=elev,
            azim=azim,
        )


def plot_phase_3d_interactive(
    trajectories,
    use="sal",
    tmin=PHASE_TMIN,
    outfile="corrected_coeff_phase_3d_sal_interactive.html",
    max_points_per_trajectory=None,
):
    """
    Interactive rotatable 3D phase portrait using Plotly.

    Output:
        an HTML file that can be opened in a browser.
        You can rotate, zoom, and pan with the mouse.

    Coordinates:
        x = tau11
        y = psi11

    Third coordinate:
        use="sal" -> z = s11
        use="NuT" -> z = NuT
        use="NuS" -> z = NuS
    """

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Skipping interactive 3D HTML plot.")
        print("Install it with: pip install plotly")
        return

    if use == "sal":
        zkey = "sal"
        zlabel = "s11"
        title = "Interactive 3D phase portrait: tau11, psi11, s11"
    elif use == "NuT":
        zkey = "NuT"
        zlabel = "NuT"
        title = "Interactive 3D phase portrait: tau11, psi11, NuT"
    elif use == "NuS":
        zkey = "NuS"
        zlabel = "NuS"
        title = "Interactive 3D phase portrait: tau11, psi11, NuS"
    else:
        raise ValueError("use must be 'sal', 'NuT', or 'NuS'")

    fig = go.Figure()

    for idx, tr in enumerate(trajectories, start=1):
        if len(tr["t"]) == 0:
            continue

        mask = tr["t"] >= tmin
        if not np.any(mask):
            continue

        x = tr["tau"][mask]
        y = tr["psi"][mask]
        z = tr[zkey][mask]

        if max_points_per_trajectory is not None and len(x) > max_points_per_trajectory:
            stride = int(np.ceil(len(x) / max_points_per_trajectory))
            x = x[::stride]
            y = y[::stride]
            z = z[::stride]

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(width=2),
                opacity=0.65,
                name=f"trajectory {idx}",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="tau11",
            yaxis_title="psi11",
            zaxis_title=zlabel,
            xaxis=dict(showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True),
            zaxis=dict(showgrid=True, zeroline=True),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
        width=950,
        height=850,
        margin=dict(l=0, r=0, b=0, t=55),
    )

    fig.write_html(outfile, include_plotlyjs="cdn")
    print(f"Saved interactive rotatable 3D plot: {outfile}")

    # In many IDEs this opens in the browser. If Spyder suppresses it,
    # open the saved HTML file manually.
    fig.show()



# ============================================================
# Nebula-style cosmic animation + GIF / MP4 export
# ============================================================

def _prepare_cosmic_phase_data(trajectories, use="NuT", tmin=PHASE_TMIN, max_points=2200):
    if use == "sal":
        zkey = "sal"
        zlabel = "s11"
        title = r"Cosmic 3D phase portrait: $\tau_{11}$, $\psi_{11}$, $s_{11}$"
    elif use == "NuT":
        zkey = "NuT"
        zlabel = "NuT"
        title = r"Cosmic 3D phase portrait: $\tau_{11}$, $\psi_{11}$, $Nu_T$"
    elif use == "NuS":
        zkey = "NuS"
        zlabel = "NuS"
        title = r"Cosmic 3D phase portrait: $\tau_{11}$, $\psi_{11}$, $Nu_S$"
    else:
        raise ValueError("use must be 'sal', 'NuT', or 'NuS'")

    usable = []
    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue
        mask = tr["t"] >= tmin
        if np.any(mask):
            usable.append((np.count_nonzero(mask), tr, mask))

    if not usable:
        raise RuntimeError("No trajectory data available beyond the chosen tmin.")

    usable.sort(key=lambda q: q[0], reverse=True)
    _, tr_main, mask_main = usable[0]

    x = tr_main["tau"][mask_main]
    y = tr_main["psi"][mask_main]
    z = tr_main[zkey][mask_main]

    if len(x) > max_points:
        stride = int(np.ceil(len(x) / max_points))
        x = x[::stride]
        y = y[::stride]
        z = z[::stride]

    background_data = []
    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue
        mask = tr["t"] >= tmin
        if not np.any(mask):
            continue

        xb = tr["tau"][mask]
        yb = tr["psi"][mask]
        zb = tr[zkey][mask]

        if len(xb) > 800:
            stride = int(np.ceil(len(xb) / 800))
            xb = xb[::stride]
            yb = yb[::stride]
            zb = zb[::stride]

        background_data.append((xb, yb, zb))

    return x, y, z, zlabel, title, background_data


def _make_nebula_and_star_traces(go, x, y, z, seed=12345):
    rng = np.random.default_rng(seed)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    zr = max(zmax - zmin, 1e-9)

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)

    # Stars
    star_n = 420
    xs = xmid + rng.uniform(-1.8 * xr, 1.8 * xr, star_n)
    ys = ymid + rng.uniform(-1.8 * yr, 1.8 * yr, star_n)
    zs = zmid + rng.uniform(-1.8 * zr, 1.8 * zr, star_n)
    star_sizes = rng.uniform(1.0, 3.2, star_n)
    star_brightness = rng.uniform(0.25, 1.0, star_n)

    star_colors = []
    for b in star_brightness:
        if b > 0.85:
            star_colors.append("rgba(255,245,200,0.95)")
        elif b > 0.65:
            star_colors.append("rgba(200,235,255,0.80)")
        else:
            star_colors.append("rgba(180,180,255,0.45)")

    traces = [
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=star_sizes, color=star_colors, opacity=1.0),
            hoverinfo="skip",
            showlegend=False,
        )
    ]

    cloud_specs = [
        (-0.70,  0.95,  0.20, 0.45, 0.32, 0.28, "rgba(104,100,255,0.11)"),
        ( 0.95, -0.85, -0.10, 0.42, 0.28, 0.25, "rgba(255, 80,190,0.10)"),
        ( 0.05,  0.15,  0.85, 0.34, 0.34, 0.24, "rgba( 40,220,255,0.08)"),
        (-0.25, -0.40,  0.00, 0.30, 0.24, 0.20, "rgba(255,200, 80,0.06)"),
    ]

    for cx_f, cy_f, cz_f, sx_f, sy_f, sz_f, color in cloud_specs:
        npts = 800
        cx = xmid + cx_f * xr
        cy = ymid + cy_f * yr
        cz = zmid + cz_f * zr

        xs = cx + rng.normal(scale=sx_f * xr, size=npts)
        ys = cy + rng.normal(scale=sy_f * yr, size=npts)
        zs = cz + rng.normal(scale=sz_f * zr, size=npts)
        sizes = rng.uniform(4.0, 12.0, size=npts)

        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(size=sizes, color=color, opacity=1.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


def _make_cosmic_base_layout(go, title, zlabel):
    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgba(8,10,22,1.0)",
        showgrid=True,
        gridcolor="rgba(170,220,255,0.09)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.12)",
        showspikes=False,
        color="#d7e6ff",
        tickfont=dict(color="#cfe3ff", size=12),
    )

    return go.Layout(
        title=dict(text=title, font=dict(size=24, color="#f4f8ff")),
        paper_bgcolor="#02040a",
        plot_bgcolor="#02040a",
        width=980,
        height=860,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(
                axis_style,
                title=dict(text="tau11", font=dict(color="#f4f8ff", size=18)),
            ),
            yaxis=dict(
                axis_style,
                title=dict(text="psi11", font=dict(color="#f4f8ff", size=18)),
            ),
            zaxis=dict(
                axis_style,
                title=dict(text=zlabel, font=dict(color="#f4f8ff", size=18)),
            ),
            bgcolor="rgba(2,4,10,1.0)",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.65, y=1.20, z=0.85)),
        ),
        annotations=[
            dict(
                text="artistic dark animation • drag to rotate, scroll to zoom",
                x=0.99, y=0.985, xref="paper", yref="paper",
                xanchor="right", yanchor="top", showarrow=False,
                font=dict(size=12, color="#9ecbff"),
            )
        ],
    )


def plot_phase_3d_cosmic_nebula_animation(
    trajectories,
    use="NuT",
    tmin=PHASE_TMIN,
    outfile="corrected_coeff_phase_3d_nebula_cosmic.html",
    max_points=2200,
    tail_fraction=0.18,
    nframes=160,
    camera_radius=1.85,
    camera_z=0.88,
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Install it with: pip install plotly")
        return

    x, y, z, zlabel, title, background_data = _prepare_cosmic_phase_data(
        trajectories, use=use, tmin=tmin, max_points=max_points
    )

    cosmic_scale = [
        [0.00, "#14213d"],
        [0.10, "#2252a3"],
        [0.25, "#17c9ff"],
        [0.42, "#7b5cff"],
        [0.60, "#d946ef"],
        [0.78, "#ff5ea8"],
        [0.92, "#ffd166"],
        [1.00, "#fff7cc"],
    ]

    # No extra ribbon clutter: keep the scene clean like typical strange-attractor demos.
    background_traces = []

    nebula_traces = _make_clean_star_traces(go, x, y, z, nstars=70)

    full_trace = go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color=np.linspace(0, 1, len(x)), colorscale=cosmic_scale, width=5),
        opacity=0.20, hoverinfo="skip", showlegend=False,
    )

    tail_len = max(24, int(tail_fraction * len(x)))
    i0 = tail_len

    tail_trace = go.Scatter3d(
        x=x[:i0], y=y[:i0], z=z[:i0], mode="lines",
        line=dict(color=np.linspace(0, 1, i0), colorscale=cosmic_scale, width=9),
        hoverinfo="skip", showlegend=False,
    )

    head_trace = go.Scatter3d(
        x=[x[i0 - 1]], y=[y[i0 - 1]], z=[z[i0 - 1]], mode="markers",
        marker=dict(size=8, color="#fff7cc", line=dict(color="#ffffff", width=1), opacity=1.0),
        hoverinfo="skip", showlegend=False,
    )

    fig = go.Figure(data=nebula_traces + background_traces + [full_trace, tail_trace, head_trace])

    frame_indices = np.linspace(tail_len, len(x) - 1, nframes, dtype=int)
    frames = []

    for k, idx_pt in enumerate(frame_indices):
        i1 = max(0, idx_pt - tail_len + 1)
        xt = x[i1:idx_pt + 1]
        yt = y[i1:idx_pt + 1]
        zt = z[i1:idx_pt + 1]

        ang = 2.0 * np.pi * k / max(nframes, 1)
        eye = dict(x=camera_radius * np.cos(ang), y=camera_radius * np.sin(ang), z=camera_z)

        frame_data = []
        frame_data.extend(nebula_traces)
        frame_data.extend(background_traces)
        frame_data.append(full_trace)
        frame_data.append(
            go.Scatter3d(
                x=xt, y=yt, z=zt, mode="lines",
                line=dict(color=np.linspace(0, 1, len(xt)), colorscale=cosmic_scale, width=10),
                hoverinfo="skip", showlegend=False,
            )
        )
        frame_data.append(
            go.Scatter3d(
                x=[x[idx_pt]], y=[y[idx_pt]], z=[z[idx_pt]], mode="markers",
                marker=dict(size=9, color="#fff7cc", line=dict(color="#ffffff", width=1), opacity=1.0),
                hoverinfo="skip", showlegend=False,
            )
        )

        frames.append(
            go.Frame(
                data=frame_data,
                name=f"frame{k}",
                layout=go.Layout(scene_camera=dict(eye=eye)),
            )
        )

    fig.frames = frames
    fig.update_layout(_make_cosmic_base_layout(go, title, zlabel))
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.04, y=0.98, xanchor="left", yanchor="top", direction="left",
                bgcolor="rgba(20,24,40,0.82)",
                bordercolor="rgba(180,220,255,0.25)",
                font=dict(color="#f5f8ff", size=13),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=70, redraw=True),
                                         transition=dict(duration=0),
                                         fromcurrent=True, mode="immediate")],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           transition=dict(duration=0),
                                           mode="immediate")],
                    ),
                ],
            )
        ]
    )

    fig.write_html(outfile, include_plotlyjs="cdn")
    print(f"Saved nebula-style cosmic animated HTML: {outfile}")
    fig.show()


def export_phase_3d_cosmic_gif_mp4(
    trajectories,
    use="NuT",
    tmin=PHASE_TMIN,
    gif_out="corrected_coeff_phase_3d_nebula_cosmic.gif",
    mp4_out="corrected_coeff_phase_3d_nebula_cosmic.mp4",
    export_gif=True,
    export_mp4=True,
    max_points=1800,
    tail_fraction=0.18,
    nframes=120,
    fps=20,
    camera_radius=1.85,
    camera_z=0.88,
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Install it with: pip install plotly")
        return

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not installed. Install it with: pip install imageio pillow imageio-ffmpeg")
        return

    x, y, z, zlabel, title, background_data = _prepare_cosmic_phase_data(
        trajectories, use=use, tmin=tmin, max_points=max_points
    )

    cosmic_scale = [
        [0.00, "#14213d"],
        [0.10, "#2252a3"],
        [0.25, "#17c9ff"],
        [0.42, "#7b5cff"],
        [0.60, "#d946ef"],
        [0.78, "#ff5ea8"],
        [0.92, "#ffd166"],
        [1.00, "#fff7cc"],
    ]

    nebula_traces = _make_clean_star_traces(go, x, y, z, nstars=70)

    background_traces = []
    for xb, yb, zb in background_data:
        background_traces.append(
            go.Scatter3d(
                x=xb, y=yb, z=zb, mode="lines",
                line=dict(color="rgba(195,215,255,0.07)", width=2),
                hoverinfo="skip", showlegend=False,
            )
        )

    full_trace = go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color=np.linspace(0, 1, len(x)), colorscale=cosmic_scale, width=5),
        opacity=0.20, hoverinfo="skip", showlegend=False,
    )

    tail_len = max(24, int(tail_fraction * len(x)))
    frame_indices = np.linspace(tail_len, len(x) - 1, nframes, dtype=int)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        png_files = []

        print("Rendering nebula-style cosmic frames...")

        for k, idx_pt in enumerate(frame_indices):
            i1 = max(0, idx_pt - tail_len + 1)
            xt = x[i1:idx_pt + 1]
            yt = y[i1:idx_pt + 1]
            zt = z[i1:idx_pt + 1]

            ang = 2.0 * np.pi * k / max(nframes, 1)
            eye = dict(x=camera_radius * np.cos(ang), y=camera_radius * np.sin(ang), z=camera_z)

            frame_tail = go.Scatter3d(
                x=xt, y=yt, z=zt, mode="lines",
                line=dict(color=np.linspace(0, 1, len(xt)), colorscale=cosmic_scale, width=10),
                hoverinfo="skip", showlegend=False,
            )

            frame_head = go.Scatter3d(
                x=[x[idx_pt]], y=[y[idx_pt]], z=[z[idx_pt]], mode="markers",
                marker=dict(size=9, color="#fff7cc", line=dict(color="#ffffff", width=1), opacity=1.0),
                hoverinfo="skip", showlegend=False,
            )

            fig = go.Figure(data=nebula_traces + background_traces + [full_trace, frame_tail, frame_head])
            fig.update_layout(_make_cosmic_base_layout(go, title, zlabel))
            fig.update_layout(scene_camera=dict(eye=eye))

            png_path = tmpdir / f"frame_{k:04d}.png"
            try:
                fig.write_image(str(png_path), scale=1.4)
            except Exception as e:
                print("Could not write frame images.")
                print("Please install or upgrade kaleido:")
                print("    pip install -U kaleido")
                print("Original error:", e)
                return

            png_files.append(png_path)

            if (k + 1) % 10 == 0 or (k + 1) == len(frame_indices):
                print(f"  rendered {k+1}/{len(frame_indices)} frames")

        if export_gif:
            print(f"Writing GIF: {gif_out}")
            images = [imageio.imread(p) for p in png_files]
            imageio.mimsave(gif_out, images, fps=fps, loop=0)
            print(f"Saved GIF: {gif_out}")

        if export_mp4:
            print(f"Writing MP4: {mp4_out}")
            with imageio.get_writer(mp4_out, fps=fps, codec="libx264", quality=8) as writer:
                for p in png_files:
                    writer.append_data(imageio.imread(p))
            print(f"Saved MP4: {mp4_out}")



# ============================================================
# Cleaner artistic dark visualizations
# ============================================================

def _make_clean_star_traces(go, x, y, z, seed=12345, nstars=70):
    """
    Create a very sparse star field only.
    This replaces the earlier noisy nebula point clouds.
    """

    rng = np.random.default_rng(seed)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    zr = max(zmax - zmin, 1e-9)

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)

    xs = xmid + rng.uniform(-1.9 * xr, 1.9 * xr, nstars)
    ys = ymid + rng.uniform(-1.9 * yr, 1.9 * yr, nstars)
    zs = zmid + rng.uniform(-1.9 * zr, 1.9 * zr, nstars)

    sizes = rng.uniform(0.7, 2.2, nstars)
    colors = []
    for _ in range(nstars):
        u = rng.random()
        if u > 0.88:
            colors.append("rgba(255,245,210,0.85)")
        elif u > 0.55:
            colors.append("rgba(190,225,255,0.40)")
        else:
            colors.append("rgba(170,180,255,0.20)")

    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=1.0),
            hoverinfo="skip",
            showlegend=False,
        )
    ]




def save_lorenz_style_3d_projected_png(
    trajectories,
    use="sal",
    tmin=PHASE_TMIN,
    outfile="corrected_coeff_phase_3d_sal_lorenz_style.png",
    elev=18.0,
    azim=35.0,
    max_points_per_traj=1800,
    target_aspect=1.12,
):
    """
    Save a dark artistic still image using a 3D trajectory embedding projected
    to 2D with a chosen viewing angle, but rendered in a thin-line strange-attractor-demo style.

    Typical use here:
        use="sal"  -> coordinates (tau11, psi11, s11)

    Improvements in this version:
      - less horizontally stretched
      - automatic aspect-ratio correction so the final silhouette is closer
        to the regular scientific 3D view
      - thin luminous lines with subtle glow
    """

    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    usable = []
    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue
        mask = tr["t"] >= tmin
        if np.any(mask):
            usable.append(tr)

    if not usable:
        print("No trajectory data available beyond tmin for artistic 3D projection.")
        return

    # Collect all projected trajectories first so framing/aspect correction
    # is computed globally rather than trajectory-by-trajectory.
    projected = []

    def _scale(v):
        v = np.asarray(v, dtype=float)
        m = np.mean(v)
        s = 0.5 * (np.max(v) - np.min(v))
        if s <= 1e-12:
            s = 1.0
        return (v - m) / s

    az = np.deg2rad(azim)
    el = np.deg2rad(elev)

    Rz = np.array([
        [ np.cos(az), -np.sin(az), 0.0],
        [ np.sin(az),  np.cos(az), 0.0],
        [        0.0,         0.0, 1.0],
    ])

    Rx = np.array([
        [1.0,       0.0,        0.0],
        [0.0,  np.cos(el), -np.sin(el)],
        [0.0,  np.sin(el),  np.cos(el)],
    ])

    for tr in usable:
        mask = tr["t"] >= tmin

        x = tr["tau"][mask]
        y = tr["psi"][mask]

        if use == "sal":
            z = tr["sal"][mask]
        elif use == "NuT":
            z = tr["NuT"][mask]
        elif use == "NuS":
            z = tr["NuS"][mask]
        else:
            raise ValueError("use must be 'sal', 'NuT', or 'NuS'")

        if len(x) > max_points_per_traj:
            stride = int(np.ceil(len(x) / max_points_per_traj))
            x = x[::stride]
            y = y[::stride]
            z = z[::stride]

        X = _scale(x)
        Y = _scale(y)
        Z = _scale(z)

        P = np.column_stack([X, Y, Z])
        Pr = P @ Rz.T @ Rx.T

        u = Pr[:, 0]
        v = Pr[:, 1]
        projected.append([u, v])

    # Global bounds before aspect correction.
    all_u = np.concatenate([uv[0] for uv in projected])
    all_v = np.concatenate([uv[1] for uv in projected])

    width = max(np.max(all_u) - np.min(all_u), 1e-12)
    height = max(np.max(all_v) - np.min(all_v), 1e-12)
    raw_aspect = width / height

    # If the projection is too elongated horizontally, stretch the vertical
    # coordinate about its global center to make the silhouette closer to the
    # standard 3D scientific view.
    if raw_aspect > target_aspect:
        v_center = 0.5 * (np.max(all_v) + np.min(all_v))
        stretch = raw_aspect / target_aspect
        for uv in projected:
            uv[1] = v_center + stretch * (uv[1] - v_center)

    # Recompute bounds after aspect correction.
    all_u = np.concatenate([uv[0] for uv in projected])
    all_v = np.concatenate([uv[1] for uv in projected])

    cmap = LinearSegmentedColormap.from_list(
        "lorenzish_thin",
        [
            "#2f1d57",  # deep violet
            "#4154a7",  # blue
            "#2b8dc4",  # azure
            "#30c0be",  # cyan-turquoise
            "#6cd36b",  # green
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    for (u, v) in projected:
        if len(u) < 3:
            continue

        pts = np.column_stack([u, v])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        c = np.linspace(0.0, 1.0, len(segs))

        # Thin subtle glow.
        for lw, alpha in [(8.0, 0.010), (5.0, 0.020), (2.6, 0.050)]:
            lc = LineCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha)
            lc.set_array(c)
            ax.add_collection(lc)

        # Thin luminous line.
        lc_main = LineCollection(segs, cmap=cmap, linewidths=1.05, alpha=0.80)
        lc_main.set_array(c)
        ax.add_collection(lc_main)

        # Very thin dark center line for that strange-attractor-demo look.
        lc_core = LineCollection(segs, colors="black", linewidths=0.55, alpha=0.92)
        ax.add_collection(lc_core)

    padx = 0.04 * max(np.max(all_u) - np.min(all_u), 1e-12)
    pady = 0.04 * max(np.max(all_v) - np.min(all_v), 1e-12)

    ax.set_xlim(np.min(all_u) - padx, np.max(all_u) + padx)
    ax.set_ylim(np.min(all_v) - pady, np.max(all_v) + pady)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(outfile, dpi=240, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.show()

    print(f"Saved artistic 3D-projected still image: {outfile}")





def save_lorenz_style_stationary_png(
    trajectories,
    tmin=PHASE_TMIN,
    outfile="corrected_coeff_phase_lorenz_style_still.png",
):
    """
    Save a dark artistic still image inspired by common strange-attractor renders.

    Projection:
        x = tau11
        y = psi11

    Style:
        - black background
        - no axes
        - layered luminous strokes
        - smooth color transition: violet -> blue -> cyan -> green
    """

    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    usable = []
    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue
        mask = tr["t"] >= tmin
        if np.any(mask):
            usable.append((np.count_nonzero(mask), tr, mask))

    if not usable:
        print("No trajectory data available beyond tmin for artistic still.")
        return

    usable.sort(key=lambda q: q[0], reverse=True)
    _, tr_main, mask = usable[0]

    x = tr_main["tau"][mask]
    y = tr_main["psi"][mask]

    # Downsample a bit for faster rendering while keeping a smooth look.
    max_points = 3500
    if len(x) > max_points:
        stride = int(np.ceil(len(x) / max_points))
        x = x[::stride]
        y = y[::stride]

    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    c = np.linspace(0.0, 1.0, len(segs))

    cmap = LinearSegmentedColormap.from_list(
        "lorenzish",
        [
            "#33205e",  # deep violet
            "#455cb3",  # blue
            "#2aa7d7",  # cyan-blue
            "#35c6be",  # turquoise
            "#68d06e",  # green
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    # Several blurred glow layers.
    for lw, alpha in [(34, 0.018), (24, 0.028), (16, 0.050), (10, 0.085)]:
        lc = LineCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha)
        lc.set_array(c)
        ax.add_collection(lc)

    # Main luminous ribbon.
    lc_main = LineCollection(segs, cmap=cmap, linewidths=3.8, alpha=0.96)
    lc_main.set_array(c)
    ax.add_collection(lc_main)

    # Dark central core for the strange-attractor-demo look.
    lc_core = LineCollection(segs, colors="black", linewidths=1.6, alpha=0.88)
    ax.add_collection(lc_core)

    padx = 0.035 * max(np.max(x) - np.min(x), 1e-12)
    pady = 0.035 * max(np.max(y) - np.min(y), 1e-12)
    ax.set_xlim(np.min(x) - padx, np.max(x) + padx)
    ax.set_ylim(np.min(y) - pady, np.max(y) + pady)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(outfile, dpi=220, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.show()

    print(f"Saved artistic still image: {outfile}")



def save_lorenz_style_3d_projected_electrons_animation(
    trajectories,
    use="sal",
    tmin=PHASE_TMIN,
    gif_out="corrected_coeff_phase_3d_sal_lorenz_electrons.gif",
    mp4_out="corrected_coeff_phase_3d_sal_lorenz_electrons.mp4",
    export_gif=True,
    export_mp4=False,
    elev=18.0,
    azim=35.0,
    max_points_per_traj=1800,
    target_aspect=1.12,
    nframes=180,
    fps=24,
    nparticles=14,
):
    """
    Make an animation with bright "electrons" moving along the projected
    phase trajectories.

    Visual idea:
      - dark background
      - thin artistic phase lines
      - small glowing particles moving along the trajectories
      - one GIF and/or MP4 export

    This is based on the same projected 3D embedding used for the artistic still:
        (tau11, psi11, s11)  -> rotated 3D view -> 2D projection
    """

    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    usable = []
    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue
        mask = tr["t"] >= tmin
        if np.any(mask):
            usable.append(tr)

    if not usable:
        print("No trajectory data available beyond tmin for electron animation.")
        return

    def _scale(v):
        v = np.asarray(v, dtype=float)
        m = np.mean(v)
        s = 0.5 * (np.max(v) - np.min(v))
        if s <= 1e-12:
            s = 1.0
        return (v - m) / s

    az = np.deg2rad(azim)
    el = np.deg2rad(elev)

    Rz = np.array([
        [ np.cos(az), -np.sin(az), 0.0],
        [ np.sin(az),  np.cos(az), 0.0],
        [        0.0,         0.0, 1.0],
    ])

    Rx = np.array([
        [1.0,       0.0,        0.0],
        [0.0,  np.cos(el), -np.sin(el)],
        [0.0,  np.sin(el),  np.cos(el)],
    ])

    projected = []

    for tr in usable:
        mask = tr["t"] >= tmin

        x = tr["tau"][mask]
        y = tr["psi"][mask]

        if use == "sal":
            z = tr["sal"][mask]
        elif use == "NuT":
            z = tr["NuT"][mask]
        elif use == "NuS":
            z = tr["NuS"][mask]
        else:
            raise ValueError("use must be 'sal', 'NuT', or 'NuS'")

        if len(x) > max_points_per_traj:
            stride = int(np.ceil(len(x) / max_points_per_traj))
            x = x[::stride]
            y = y[::stride]
            z = z[::stride]

        X = _scale(x)
        Y = _scale(y)
        Z = _scale(z)

        P = np.column_stack([X, Y, Z])
        Pr = P @ Rz.T @ Rx.T

        u = Pr[:, 0]
        v = Pr[:, 1]
        projected.append([u, v])

    # Global aspect correction: same idea as in the artistic still.
    all_u = np.concatenate([uv[0] for uv in projected])
    all_v = np.concatenate([uv[1] for uv in projected])

    width = max(np.max(all_u) - np.min(all_u), 1e-12)
    height = max(np.max(all_v) - np.min(all_v), 1e-12)
    raw_aspect = width / height

    if raw_aspect > target_aspect:
        v_center = 0.5 * (np.max(all_v) + np.min(all_v))
        stretch = raw_aspect / target_aspect
        for uv in projected:
            uv[1] = v_center + stretch * (uv[1] - v_center)

    all_u = np.concatenate([uv[0] for uv in projected])
    all_v = np.concatenate([uv[1] for uv in projected])

    cmap = LinearSegmentedColormap.from_list(
        "lorenzish_thin",
        [
            "#2f1d57",  # deep violet
            "#4154a7",  # blue
            "#2b8dc4",  # azure
            "#30c0be",  # cyan-turquoise
            "#6cd36b",  # green
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    # Background phase trajectories
    for (u, v) in projected:
        if len(u) < 3:
            continue

        pts = np.column_stack([u, v])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        c = np.linspace(0.0, 1.0, len(segs))

        for lw, alpha in [(8.0, 0.008), (5.0, 0.016), (2.6, 0.040)]:
            lc = LineCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha)
            lc.set_array(c)
            ax.add_collection(lc)

        lc_main = LineCollection(segs, cmap=cmap, linewidths=0.95, alpha=0.70)
        lc_main.set_array(c)
        ax.add_collection(lc_main)

        lc_core = LineCollection(segs, colors="black", linewidths=0.50, alpha=0.90)
        ax.add_collection(lc_core)

    padx = 0.04 * max(np.max(all_u) - np.min(all_u), 1e-12)
    pady = 0.04 * max(np.max(all_v) - np.min(all_v), 1e-12)

    ax.set_xlim(np.min(all_u) - padx, np.max(all_u) + padx)
    ax.set_ylim(np.min(all_v) - pady, np.max(all_v) + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Choose trajectories for animated particles.
    good = [uv for uv in projected if len(uv[0]) > 50]
    if len(good) == 0:
        print("Projected trajectories are too short for electron animation.")
        return

    nparticles = min(nparticles, len(good))
    idxs = np.linspace(0, len(good) - 1, nparticles, dtype=int)
    particle_paths = [good[i] for i in idxs]

    phase_offsets = np.linspace(0.0, 1.0, nparticles, endpoint=False)
    speed_factors = np.linspace(0.9, 1.25, nparticles)

    # Particle colors: mostly cyan/green/white, like glowing electrons.
    particle_colors = [
        "#b7fff6", "#9df6ff", "#7fefff", "#6ff2d7", "#8ef8c0",
        "#d5fff7", "#84ecff", "#9ffff0", "#cafcff", "#7fffe4",
        "#aef7ff", "#c4ffe8", "#8ef6ff", "#dfffea",
    ][:nparticles]

    # Glow layer + core layer.
    glow = ax.scatter(
        np.zeros(nparticles),
        np.zeros(nparticles),
        s=120,
        c=particle_colors,
        alpha=0.12,
        linewidths=0,
        zorder=10,
    )

    core = ax.scatter(
        np.zeros(nparticles),
        np.zeros(nparticles),
        s=18,
        c=particle_colors,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.30,
        zorder=11,
    )

    # Optional very short trailing tails for each particle.
    tail_lines = []
    for _ in range(nparticles):
        line, = ax.plot([], [], color="#aef7ff", lw=0.8, alpha=0.25, zorder=9)
        tail_lines.append(line)

    def _particle_xy(frame):
        xs = []
        ys = []
        trails = []

        for j, (u, v) in enumerate(particle_paths):
            n = len(u)
            # Advance around the trajectory with phase offset and mild speed variation.
            pos = (phase_offsets[j] * n + frame * speed_factors[j] * n / nframes) % n
            idx = int(pos)

            xs.append(u[idx])
            ys.append(v[idx])

            tail_len = min(18, n)
            tail_idx = [(idx - k) % n for k in range(tail_len)][::-1]
            trails.append((u[tail_idx], v[tail_idx]))

        return np.asarray(xs), np.asarray(ys), trails

    def init():
        xs, ys, trails = _particle_xy(0)
        offs = np.column_stack([xs, ys])
        glow.set_offsets(offs)
        core.set_offsets(offs)

        for line, (tx, ty) in zip(tail_lines, trails):
            line.set_data(tx, ty)

        return [glow, core, *tail_lines]

    def update(frame):
        xs, ys, trails = _particle_xy(frame)
        offs = np.column_stack([xs, ys])
        glow.set_offsets(offs)
        core.set_offsets(offs)

        for line, (tx, ty) in zip(tail_lines, trails):
            line.set_data(tx, ty)

        return [glow, core, *tail_lines]

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=nframes,
        interval=1000.0 / fps,
        blit=True,
    )

    saved_any = False

    if export_gif:
        print(f"Saving electron GIF animation: {gif_out}")
        try:
            ani.save(gif_out, writer=PillowWriter(fps=fps), dpi=200)
            print(f"Saved electron GIF animation: {gif_out}")
            saved_any = True
        except Exception as e:
            print("Could not save GIF animation.")
            print("Make sure Pillow is installed: pip install pillow")
            print("Original error:", e)

    if export_mp4:
        print(f"Saving electron MP4 animation: {mp4_out}")
        try:
            ani.save(mp4_out, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=200)
            print(f"Saved electron MP4 animation: {mp4_out}")
            saved_any = True
        except Exception as e:
            print("Could not save MP4 animation.")
            print("You may need ffmpeg available in Matplotlib / system PATH.")
            print("Original error:", e)

    if not saved_any:
        print("Electron animation was created in memory, but no file was saved.")
        print("Check MAKE_ARTISTIC_STYLE_ELECTRONS_GIF / MAKE_ARTISTIC_STYLE_ELECTRONS_MP4.")

    plt.close(fig)





# ============================================================
# LIGHTWEIGHT MOBIUS / TWISTED-RIBBON DIAGNOSTICS
# ============================================================

def run_lightweight_mobius_checks(trajectory, outfile_prefix="mobius_check"):
    """
    Practical diagnostics for a projected twisted/Mobius-like ribbon.

    This does not prove topology rigorously; it checks two useful numerical signs:
    1) the 2D projection has bins where the hidden coordinate s11 separates, and
    2) local normals along the 3D curve show strong orientation reversal/twist.
    """
    mask = trajectory["t"] >= PHASE_TMIN
    tau = trajectory["tau"][mask]
    psi = trajectory["psi"][mask]
    sal = trajectory["sal"][mask]

    if len(tau) < 200:
        print("Mobius check skipped: not enough post-transient points.")
        return

    bins = 90
    H, xe, ye = np.histogram2d(tau, psi, bins=bins)
    smin = np.full_like(H, np.inf, dtype=float)
    smax = np.full_like(H, -np.inf, dtype=float)
    ix = np.clip(np.searchsorted(xe, tau, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, psi, side="right") - 1, 0, bins - 1)
    for i, j, s in zip(ix, iy, sal):
        if s < smin[i, j]:
            smin[i, j] = s
        if s > smax[i, j]:
            smax[i, j] = s
    valid = H >= 8
    sep = np.where(valid, smax - smin, np.nan)
    max_sep = float(np.nanmax(sep)) if np.any(valid) else float("nan")

    # Local normal proxy from derivatives of 3D curve.
    pts = np.column_stack([tau, psi, sal])
    stride = max(1, len(pts) // 3500)
    pts = pts[::stride]
    tangent = np.gradient(pts, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-14)
    ref = np.array([0.0, 0.0, 1.0])
    normals = np.cross(tangent, ref)
    bad = np.linalg.norm(normals, axis=1) < 1e-10
    normals[bad] = np.cross(tangent[bad], np.array([0.0, 1.0, 0.0]))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-14)
    dot0 = np.sum(normals[0] * normals, axis=1)
    min_dot = float(np.nanmin(dot0))
    end_dot = float(np.sum(normals[0] * normals[-1]))

    # Histogram of hidden-coordinate separation.
    vals = sep[np.isfinite(sep)]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    if len(vals):
        ax.hist(vals, bins=40, color="black", alpha=0.85)
    style_normal_axes(ax)
    ax.set_xlabel(r"local $s_{11}$ separation in projected $(\tau_{11},\psi_{11})$ bin", fontsize=12)
    ax.set_ylabel("count", fontsize=12)
    ax.set_title("Projected-overlap separation check", fontsize=15)
    plt.tight_layout()
    plt.savefig(outfile_prefix + "_overlap_separation.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    ax.plot(dot0, "k-", lw=0.8)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axhline(-1.0, color="0.7", lw=0.8, ls="--")
    style_normal_axes(ax)
    ax.set_xlabel("sample index along asymptotic curve", fontsize=12)
    ax.set_ylabel("normal dot initial normal", fontsize=12)
    ax.set_title("Local normal orientation/twist check", fontsize=15)
    plt.tight_layout()
    plt.savefig(outfile_prefix + "_normal_dot.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    with open(outfile_prefix + "_report.txt", "w", encoding="utf-8") as f:
        f.write("Lightweight Mobius / twisted-ribbon diagnostic\n")
        f.write(f"RT={RT:g}, RS={RS:g}, Pr={SIGMA:g}, k={KAPPA:.8g}\n")
        f.write(f"section tau11={TAU_SECTION:g}\n")
        f.write(f"post-transient points={len(tau)}\n")
        f.write(f"max projected-bin s11 separation={max_sep:.8g}\n")
        f.write(f"minimum normal dot initial normal={min_dot:.8g}\n")
        f.write(f"end normal dot initial normal={end_dot:.8g}\n")
        f.write("Interpretation: large hidden-coordinate separation plus strongly negative normal-dot values supports a projected twisted ribbon, but is not a formal topological proof.\n")

    print(f"Saved Mobius/twisted-ribbon checks with prefix: {outfile_prefix}")

# ============================================================
# MAIN
# ============================================================

def main():
    ops = load_operators()

    print()
    print("=" * 90)
    print("PARAMETERS")
    print("=" * 90)
    print(f"N                 = {N}")
    print(f"2N cutoff          = {2*N}")
    print(f"RT                = {RT}")
    print(f"RS                = {RS}")
    print(f"SIGMA/Pr          = {SIGMA}")
    print(f"KAPPA             = {KAPPA}")
    print(f"DT                = {DT}")
    print(f"T_TOTAL           = {T_TOTAL}")
    print(f"N_TRAJ            = {N_TRAJ}")
    print(f"CACHE_FILE        = {CACHE_FILE}")
    print(f"Interactive 3D    = {MAKE_INTERACTIVE_3D_HTML}")
    if MAKE_INTERACTIVE_3D_HTML:
        print(f"3D salinity HTML  = {OUT_PHASE_3D_SAL_HTML}")
        print(f"3D Nusselt HTML   = {OUT_PHASE_3D_NUT_HTML}")
    print("Scientific sal 3D  = corrected_coeff_phase_3d_sal_view2.png")
    print(f"Artistic 3D still = {OUT_ARTISTIC_STYLE_3D_SAL}")
    print(f"Electron GIF      = {OUT_ARTISTIC_STYLE_ELECTRONS_GIF}")
    print(f"Electron MP4      = {OUT_ARTISTIC_STYLE_ELECTRONS_MP4}")
    print("=" * 90)
    print()

    # Warm-up compile.
    print("Compiling coefficient-space RK4 kernel...")
    y_dummy = build_base_ic()

    _ = rk4_step_coeff(
        y_dummy,
        DT,
        SIGMA,
        RT,
        RS,
        KAPPA,
        ops["lam_psi"],
        ops["lam_sc"],
        ops["Qpsi"],
        ops["Qtau"],
        ops["Qsal"],
        ops["Bpsi_sc"],
        ops["Fsc_psi"],
    )

    print("Compilation done.")

    base_y0 = build_base_ic()

    trajectories = []
    t_all = time.time()

    for traj_id in range(N_TRAJ):
        y0 = build_traj_ic(base_y0, traj_id)
        tr = integrate_one_trajectory(y0, ops, traj_id=traj_id)
        trajectories.append(tr)

    print(f"All trajectories done in {time.time() - t_all:.2f} s")

    for i, tr in enumerate(trajectories):
        if len(tr["t"]) == 0:
            print(f"Trajectory {i+1}: no saved data")
            continue

        print(f"\nTrajectory {i+1}")
        print(f"  blown        = {tr['blown']}")
        print(f"  saved points = {len(tr['t'])}")
        print(f"  tau11 range  = {tr['tau'].min():.6f} .. {tr['tau'].max():.6f}")
        print(f"  psi11 range  = {tr['psi'].min():.6f} .. {tr['psi'].max():.6f}")
        print(f"  NuT range    = {tr['NuT'].min():.6f} .. {tr['NuT'].max():.6f}")
        print(f"  NuS range    = {tr['NuS'].min():.6f} .. {tr['NuS'].max():.6f}")

    # --------------------------------------------------------
    # Phase portrait: tau11 vs psi11.
    # --------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor="white")
    ax1.set_facecolor("white")

    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue

        mask = tr["t"] >= PHASE_TMIN

        if np.any(mask):
            ax1.plot(
                tr["tau"][mask],
                tr["psi"][mask],
                color="black",
                lw=0.5,
                alpha=0.75,
            )

    ax1.set_xlim(-0.45, 0.45)
    ax1.set_ylim(-8.5, 8.5)

    style_center_axes(
        ax1,
        xticks=np.arange(-0.4, 0.41, 0.1),
        yticks=np.arange(-8, 9, 2),
    )

    # Clean paper-style labels.
    # Do NOT use ax.set_xlabel()/ax.set_ylabel() here, because centered spines
    # put those labels near the middle and can duplicate/shift them.
    ax1.text(
        0.50,
        0.955,
        r"$\psi_{11}$",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=24,
        color="black",
    )

    ax1.text(
        0.965,
        0.515,
        r"$\tau_{11}$",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=24,
        color="black",
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_title("Approximation by 8 harmonics", fontsize=26, pad=18)

    fig1.text(
        0.24,
        0.055,
        fr"$R_T = {RT:g} \qquad R_S = {RS:g} \qquad Pr = {SIGMA:g}$",
        fontsize=17,
    )

    plt.tight_layout(rect=[0.03, 0.08, 0.98, 0.96])
    plt.savefig(OUT_PHASE, dpi=300, facecolor="white", bbox_inches="tight")
    plt.show()

    print(f"Saved: {OUT_PHASE}")

    # --------------------------------------------------------
    # 3D phase portraits.
    # These are the key plots for checking whether the apparent
    # 2D self-overlap is a projected twisted/Möbius-like band.
    # --------------------------------------------------------
    # Keep the scientifically useful NuT 3D view.
    plot_phase_3d(
        trajectories,
        use="NuT",
        tmin=PHASE_TMIN,
        outfile="corrected_coeff_phase_3d_NuT.png",
        elev=24,
        azim=-58,
    )

    # Keep the standard scientific 3D view for (tau11, psi11, s11).
    plot_phase_3d(
        trajectories,
        use="sal",
        tmin=PHASE_TMIN,
        outfile="corrected_coeff_phase_3d_sal_view2.png",
        elev=18,
        azim=35,
    )

    # Lightweight Mobius / twisted-ribbon diagnostics for the reference trajectory.
    try:
        run_lightweight_mobius_checks(trajectories[0], outfile_prefix="mobius_check")
    except Exception as exc:
        print(f"Mobius check failed but integration results are still usable: {exc}")

    # Add one extra artistic view similar in spirit to common strange-attractor images.
    # Do NOT produce the old 2D-only artistic still, which was too narrow.
    if MAKE_ARTISTIC_STYLE_STILL:
        save_lorenz_style_3d_projected_png(
            trajectories,
            use="sal",
            tmin=PHASE_TMIN,
            outfile=OUT_ARTISTIC_STYLE_3D_SAL,
            elev=18,
            azim=35,
        )

    # Animation: glowing "electrons" moving along the phase trajectories.
    if MAKE_ARTISTIC_STYLE_ELECTRONS_GIF or MAKE_ARTISTIC_STYLE_ELECTRONS_MP4:
        save_lorenz_style_3d_projected_electrons_animation(
            trajectories,
            use="sal",
            tmin=PHASE_TMIN,
            gif_out=OUT_ARTISTIC_STYLE_ELECTRONS_GIF,
            mp4_out=OUT_ARTISTIC_STYLE_ELECTRONS_MP4,
            export_gif=MAKE_ARTISTIC_STYLE_ELECTRONS_GIF,
            export_mp4=MAKE_ARTISTIC_STYLE_ELECTRONS_MP4,
            elev=18,
            azim=35,
            nframes=180,
            fps=24,
            nparticles=14,
        )

    # --------------------------------------------------------
    # Interactive rotatable 3D HTML plots.
    # Open these HTML files in a browser and rotate with mouse.
    # --------------------------------------------------------
    if MAKE_INTERACTIVE_3D_HTML:
        plot_phase_3d_interactive(
            trajectories,
            use="sal",
            tmin=PHASE_TMIN,
            outfile=OUT_PHASE_3D_SAL_HTML,
            max_points_per_trajectory=3000,
        )

        plot_phase_3d_interactive(
            trajectories,
            use="NuT",
            tmin=PHASE_TMIN,
            outfile=OUT_PHASE_3D_NUT_HTML,
            max_points_per_trajectory=3000,
        )

    # --------------------------------------------------------
    # Dramatic nebula-style artistic animation.
    # --------------------------------------------------------
    if MAKE_COSMIC_NEBULA_HTML:
        plot_phase_3d_cosmic_nebula_animation(
            trajectories,
            use="NuT",
            tmin=PHASE_TMIN,
            outfile=OUT_COSMIC_NEBULA_HTML,
            max_points=2200,
            tail_fraction=0.18,
            nframes=160,
        )

    if MAKE_COSMIC_GIF or MAKE_COSMIC_MP4:
        export_phase_3d_cosmic_gif_mp4(
            trajectories,
            use="NuT",
            tmin=PHASE_TMIN,
            gif_out=OUT_COSMIC_GIF,
            mp4_out=OUT_COSMIC_MP4,
            export_gif=MAKE_COSMIC_GIF,
            export_mp4=MAKE_COSMIC_MP4,
            max_points=1800,
            tail_fraction=0.18,
            nframes=120,
            fps=20,
        )

    # --------------------------------------------------------
    # Long unperturbed reference trajectory.
    # For the paper-reproduction defaults N_TRAJ=1 and the only stored
    # trajectory is already the exact paper trajectory through t=300.
    # Reuse it instead of integrating the same IVP twice.
    # --------------------------------------------------------
    if USE_STEP_LEVEL_POINCARE_EVENTS:
        tr0, step_events = integrate_reference_with_step_events(base_y0.copy(), ops)
        print("Using step-level event trajectory for Nusselt and Poincare plots")
    elif N_TRAJ == 1 and T_TOTAL >= REF_T_TOTAL:
        tr0 = trajectories[0]
        step_events = None
        print("Using the single paper-IC trajectory for Nusselt and Poincare plots")
    else:
        tr0 = integrate_reference_trajectory(base_y0.copy(), ops)
        step_events = None
        print("Using separate unperturbed reference trajectory for Nusselt and Poincare plots")

    # --------------------------------------------------------
    # Nusselt plot.
    # --------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax2.set_facecolor("white")

    mask_nu = (tr0["t"] >= NUSS_TMIN) & (tr0["t"] <= NUSS_TMAX)

    tt = tr0["t"][mask_nu]
    nuT = tr0["NuT"][mask_nu]
    nuS = tr0["NuS"][mask_nu]

    ax2.plot(tt, nuT, color="black", lw=1.0)
    ax2.plot(tt, nuS, color="black", lw=0.75, alpha=0.55)

    ax2.set_xlim(NUSS_TMIN, NUSS_TMAX)
    if NUSS_YLIM is not None:
        ax2.set_ylim(*NUSS_YLIM)
    elif len(tt) > 0:
        ymin = min(np.min(nuT), np.min(nuS))
        ymax = max(np.max(nuT), np.max(nuS))
        dy = max(ymax - ymin, 1e-6)
        ax2.set_ylim(ymin - 0.08 * dy, ymax + 0.08 * dy)

    ax2.set_title("Time history of boundary fluxes", fontsize=24, pad=14)

    # Use only normal axis labels here. No extra duplicated annotations.
    ax2.set_xlabel(r"$t$", fontsize=20, labelpad=10)
    ax2.set_ylabel(r"$Nu$", fontsize=20, labelpad=14)

    style_normal_axes(
        ax2,
        xticks=np.linspace(NUSS_TMIN, NUSS_TMAX, 6),
    )

    if len(tt) > 0:
        # Put labels at the peaks of the corresponding curves so it is
        # visually obvious which curve is Nu_T and which is Nu_S.
        jT = int(np.argmax(nuT))
        jS = int(np.argmax(nuS))

        # Lower curve: label slightly above the peak.
        ax2.text(
            tt[jT],
            nuT[jT] + 0.10,
            r"$Nu_T$",
            fontsize=18,
            color="black",
            ha="center",
            va="bottom",
        )

        # Upper curve: label slightly below the peak so it does not go
        # outside the plotting window.
        ax2.text(
            tt[jS],
            nuS[jS] - 0.14,
            r"$Nu_S$",
            fontsize=18,
            color="black",
            ha="center",
            va="top",
        )

    fig2.text(
        0.24,
        0.05,
        fr"$R_T = {RT:g} \qquad R_S = {RS:g} \qquad Pr = {SIGMA:g}$",
        fontsize=17,
    )

    plt.tight_layout(rect=[0.03, 0.04, 0.98, 0.96])
    plt.savefig(OUT_NU, dpi=300, facecolor="white", bbox_inches="tight")
    plt.show()

    print(f"Saved: {OUT_NU}")

    # --------------------------------------------------------
    # Paper-style Poincare succession and section-based Nusselt plot.
    # --------------------------------------------------------
    if step_events is not None:
        tsec = step_events["t"]
        psic = step_events["psi"]
        nuTc = step_events["NuT"]
        nuSc = step_events["NuS"]
        dirc = step_events["direction"]
        save_events_csv(step_events, OUT_POINCARE_EVENTS_CSV)
    else:
        tsec, psic, nuTc, nuSc = section_points_tau_fixed_observables(
            tr0["t"],
            tr0["tau"],
            tr0["psi"],
            tr0["NuT"],
            tr0["NuS"],
            tau_section=TAU_SECTION,
            direction=SECTION_DIRECTION,
            t_min=PHASE_TMIN,
        )
        dirc = np.zeros_like(psic, dtype=np.int64)

    print(f"Poincare crossings after t={PHASE_TMIN:g}: {len(psic)}")

    base_title = (
        fr"Poincaré map at $\tau_{{11}}={TAU_SECTION:g}$"
        + "\n"
        + fr"RT={RT:g}, RS={RS:g}, Pr={SIGMA:g}, k={KAPPA:.5g}, crossings={len(psic)}"
    )

    # Literal time-ordered succession using all detected events.
    plot_poincare_from_event_sequence(
        psic,
        title=base_title,
        outfile=OUT_POINCARE_REF,
    )

    # Directional companion maps.
    if len(psic) > 0 and np.any(dirc == +1):
        psi_up = psic[dirc == +1]
        plot_poincare_from_event_sequence(
            psi_up,
            title=(
                fr"Upward-only Poincaré map at $\tau_{{11}}={TAU_SECTION:g}$"
                + "\n"
                + fr"RT={RT:g}, RS={RS:g}, Pr={SIGMA:g}, k={KAPPA:.5g}, crossings={len(psi_up)}"
            ),
            outfile=OUT_POINCARE_REF_UP,
        )

    if len(psic) > 0 and np.any(dirc == -1):
        psi_down = psic[dirc == -1]
        plot_poincare_from_event_sequence(
            psi_down,
            title=(
                fr"Downward-only Poincaré map at $\tau_{{11}}={TAU_SECTION:g}$"
                + "\n"
                + fr"RT={RT:g}, RS={RS:g}, Pr={SIGMA:g}, k={KAPPA:.5g}, crossings={len(psi_down)}"
            ),
            outfile=OUT_POINCARE_REF_DOWN,
        )

    # Section VI also shows cross sections in the plane (Nu, psi_11),
    # not only Nu(t). Save this separately so it is not confused with
    # a time-history plot.
    fig4, ax4 = plt.subplots(figsize=(7.2, 5.8), facecolor="white")
    ax4.set_facecolor("white")
    if len(psic) > 0:
        ax4.plot(psic, nuTc, "k.", ms=2.5, label=r"$Nu_T$")
        ax4.plot(psic, nuSc, ".", color="black", alpha=0.45, ms=2.5, label=r"$Nu_S$")
        xmin, xmax = np.min(psic), np.max(psic)
        ymin, ymax = min(np.min(nuTc), np.min(nuSc)), max(np.max(nuTc), np.max(nuSc))
        dxp = max(xmax - xmin, 1e-4)
        dyp = max(ymax - ymin, 1e-4)
        ax4.set_xlim(xmin - 0.08 * dxp, xmax + 0.08 * dxp)
        if NUSS_YLIM is not None:
            ax4.set_ylim(*NUSS_YLIM)
        else:
            ax4.set_ylim(ymin - 0.08 * dyp, ymax + 0.08 * dyp)
    style_normal_axes(ax4)
    ax4.set_xlabel(r"$\psi_{11}$", fontsize=18, labelpad=10)
    ax4.set_ylabel(r"$Nu$", fontsize=18, labelpad=12)
    ax4.set_title(
        fr"Section-based fluxes at $\tau_{{11}}={TAU_SECTION:g}$" + "\n"
        + fr"RT={RT:g}, RS={RS:g}, Pr={SIGMA:g}, k={KAPPA:.5g}",
        fontsize=16,
    )
    ax4.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    out_nu_section = "corrected_coeff_nusselts_section_tau_m0p36.png"
    plt.savefig(out_nu_section, dpi=300, facecolor="white", bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_nu_section}")


# ============================================================
# Cosmic / artistic animated 3D visualization
# ============================================================

def plot_phase_3d_cosmic_animation(
    trajectories,
    use="NuT",
    tmin=PHASE_TMIN,
    outfile="corrected_coeff_phase_3d_cosmic_animation.html",
    max_points=2200,
    tail_fraction=0.16,
    nframes=140,
    camera_radius=1.85,
    camera_z=0.82,
    show_background=True,
):
    """
    Create an animated artistic 3D HTML visualization with a dark
    'mysterious universe' look.

    What it does:
      - black / cosmic background
      - neon-colored attractor
      - moving head point
      - glowing recent tail
      - slowly rotating camera
      - Play / Pause controls in HTML

    Coordinates:
      x = tau11
      y = psi11
      z = one of: s11, NuT, NuS

    Recommended use:
      plot_phase_3d_cosmic_animation(
          trajectories,
          use="NuT",
          tmin=PHASE_TMIN,
          outfile="mobius_cosmic_NuT.html"
      )
    """

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Install it with: pip install plotly")
        return

    if use == "sal":
        zkey = "sal"
        zlabel = "s11"
        title = "Cosmic 3D phase portrait: $\\tau_{11}$, $\\psi_{11}$, $s_{11}$"
    elif use == "NuT":
        zkey = "NuT"
        zlabel = "NuT"
        title = "Cosmic 3D phase portrait: $\\tau_{11}$, $\\psi_{11}$, $Nu_T$"
    elif use == "NuS":
        zkey = "NuS"
        zlabel = "NuS"
        title = "Cosmic 3D phase portrait: $\\tau_{11}$, $\\psi_{11}$, $Nu_S$"
    else:
        raise ValueError("use must be 'sal', 'NuT', or 'NuS'")

    # --------------------------------------------------------
    # Pick the longest available post-transient trajectory as the main one.
    # This gives the cleanest moving visualization.
    # --------------------------------------------------------
    usable = []
    for tr in trajectories:
        if len(tr["t"]) == 0:
            continue
        mask = tr["t"] >= tmin
        if np.any(mask):
            usable.append((np.count_nonzero(mask), tr, mask))

    if not usable:
        print("No trajectory data available beyond tmin.")
        return

    usable.sort(key=lambda q: q[0], reverse=True)
    _, tr_main, mask_main = usable[0]

    x = tr_main["tau"][mask_main]
    y = tr_main["psi"][mask_main]
    z = tr_main[zkey][mask_main]

    if len(x) > max_points:
        stride = int(np.ceil(len(x) / max_points))
        x = x[::stride]
        y = y[::stride]
        z = z[::stride]

    # Need enough points for a good animation.
    if len(x) < 40:
        print("Not enough points to build animation.")
        return

    # Colorscale: deep-space / cosmic
    cosmic_scale = [
        [0.00, "#090b1a"],   # deep space
        [0.10, "#15244b"],   # midnight blue
        [0.22, "#234b8f"],   # electric blue
        [0.38, "#3edbf0"],   # cyan glow
        [0.55, "#8b5cf6"],   # violet
        [0.72, "#d946ef"],   # magenta
        [0.88, "#ff6ec7"],   # pink glow
        [1.00, "#fff3b0"],   # starlight
    ]

    # Background traces from all trajectories: faint silver ghosts.
    background_traces = []
    if show_background:
        for idx, tr in enumerate(trajectories, start=1):
            if len(tr["t"]) == 0:
                continue
            mask = tr["t"] >= tmin
            if not np.any(mask):
                continue

            xb = tr["tau"][mask]
            yb = tr["psi"][mask]
            zb = tr[zkey][mask]

            if len(xb) > 900:
                stride = int(np.ceil(len(xb) / 900))
                xb = xb[::stride]
                yb = yb[::stride]
                zb = zb[::stride]

            background_traces.append(
                go.Scatter3d(
                    x=xb,
                    y=yb,
                    z=zb,
                    mode="lines",
                    line=dict(color="rgba(180,200,255,0.12)", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Static full attractor in faint neon for context.
    full_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(
            color=np.linspace(0, 1, len(x)),
            colorscale=cosmic_scale,
            width=4,
        ),
        opacity=0.28,
        hoverinfo="skip",
        name="full attractor",
        showlegend=False,
    )

    # Initial tail / head.
    tail_len = max(20, int(tail_fraction * len(x)))
    i0 = tail_len

    tail_trace = go.Scatter3d(
        x=x[:i0],
        y=y[:i0],
        z=z[:i0],
        mode="lines",
        line=dict(
            color=np.linspace(0, 1, i0),
            colorscale=cosmic_scale,
            width=7,
        ),
        hoverinfo="skip",
        name="moving tail",
        showlegend=False,
    )

    head_trace = go.Scatter3d(
        x=[x[i0 - 1]],
        y=[y[i0 - 1]],
        z=[z[i0 - 1]],
        mode="markers",
        marker=dict(
            size=6,
            color="#fff3b0",
            line=dict(color="#ffffff", width=1),
            opacity=0.98,
            symbol="circle",
        ),
        hoverinfo="skip",
        name="current point",
        showlegend=False,
    )

    fig = go.Figure(data=background_traces + [full_trace, tail_trace, head_trace])

    # --------------------------------------------------------
    # Frames: moving head + glowing recent tail + slow camera rotation.
    # --------------------------------------------------------
    frame_indices = np.linspace(tail_len, len(x) - 1, nframes, dtype=int)

    xr = max(np.max(x) - np.min(x), 1e-12)
    yr = max(np.max(y) - np.min(y), 1e-12)
    zr = max(np.max(z) - np.min(z), 1e-12)

    xmid = 0.5 * (np.max(x) + np.min(x))
    ymid = 0.5 * (np.max(y) + np.min(y))
    zmid = 0.5 * (np.max(z) + np.min(z))

    frames = []
    for k, idx_pt in enumerate(frame_indices):
        i1 = max(0, idx_pt - tail_len + 1)
        xt = x[i1:idx_pt + 1]
        yt = y[i1:idx_pt + 1]
        zt = z[i1:idx_pt + 1]

        # Camera slowly rotates around the attractor.
        ang = 2.0 * np.pi * k / max(nframes, 1)
        eye = dict(
            x=camera_radius * np.cos(ang),
            y=camera_radius * np.sin(ang),
            z=camera_z,
        )

        frame_traces = []

        # Background traces remain unchanged in each frame.
        if show_background:
            for bt in background_traces:
                frame_traces.append(bt)

        frame_traces.extend([
            full_trace,
            go.Scatter3d(
                x=xt,
                y=yt,
                z=zt,
                mode="lines",
                line=dict(
                    color=np.linspace(0, 1, len(xt)),
                    colorscale=cosmic_scale,
                    width=8,
                ),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter3d(
                x=[x[idx_pt]],
                y=[y[idx_pt]],
                z=[z[idx_pt]],
                mode="markers",
                marker=dict(
                    size=7,
                    color="#fff3b0",
                    line=dict(color="#ffffff", width=1),
                    opacity=1.0,
                    symbol="circle",
                ),
                hoverinfo="skip",
                showlegend=False,
            ),
        ])

        frames.append(
            go.Frame(
                data=frame_traces,
                name=f"frame{k}",
                layout=go.Layout(
                    scene_camera=dict(eye=eye)
                ),
            )
        )

    fig.frames = frames

    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgba(18,20,35,1.0)",
        showgrid=True,
        gridcolor="rgba(180,220,255,0.12)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.18)",
        showspikes=False,
        color="#d7e6ff",
        titlefont=dict(color="#f5f8ff", size=18),
        tickfont=dict(color="#cfe3ff", size=12),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color="#f5f8ff")),
        paper_bgcolor="#05070f",
        plot_bgcolor="#05070f",
        width=980,
        height=860,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(
                axis_style,
                title=dict(text="tau11", font=dict(color="#f4f8ff", size=18)),
            ),
            yaxis=dict(
                axis_style,
                title=dict(text="psi11", font=dict(color="#f4f8ff", size=18)),
            ),
            zaxis=dict(
                axis_style,
                title=dict(text=zlabel, font=dict(color="#f4f8ff", size=18)),
            ),
            bgcolor="rgba(5,7,15,1.0)",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.65, y=1.25, z=0.85)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.04,
                y=0.98,
                xanchor="left",
                yanchor="top",
                direction="left",
                bgcolor="rgba(20,24,40,0.80)",
                bordercolor="rgba(180,220,255,0.25)",
                font=dict(color="#f5f8ff", size=13),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=70, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        annotations=[
            dict(
                text="artistic cosmic animation • drag to rotate, scroll to zoom",
                x=0.99,
                y=0.985,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=12, color="#9ecbff"),
            )
        ],
    )

    fig.write_html(outfile, include_plotlyjs="cdn")
    print(f"Saved cosmic animated 3D plot: {outfile}")
    fig.show()



if __name__ == "__main__":
    main()
