"""
Batch runner for the successful Bubnov-Galerkin double-diffusive convection cases.

This driver uses dd_convection_galerkin_coeff_solver.py as the numerical core and
adds a clean package-level workflow:

1. Paper-style phase projection: (tau_11, psi_11)
2. 3D phase checks: (tau_11, psi_11, s_11) and (tau_11, psi_11, Nu_T)
3. Step-level Poincare succession maps at tau_11 = -0.36
4. Practical Mobius-band diagnostics:
   - 3D ribbon visualization
   - projection-overlap separation in s_11
   - local-normal orientation continuity check

Default cases are intentionally modest. Add more RT values in RT_CASES if needed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import dd_convection_galerkin_coeff_solver as bg


# ============================================================
# CASE LIST
# ============================================================

# Last successful comparison set. Keep this short by default because each case
# integrates to t=300 with step-level Poincare detection.
RT_CASES = [9098.0, 9103.0, 9110.0]

# Extended set used during the last scans. Uncomment if you want the full family.
# RT_CASES = [9095.1, 9098.0, 9100.0, 9102.0, 9103.0, 9104.0, 9105.0, 9110.0, 9120.0]

RS = 8000.0
SIGMA = 1.0
KAPPA = 1.0 / math.sqrt(10.0)
TAU_SECTION = -0.36
PHASE_TMIN = 100.0
T_TOTAL = 300.0
DT = 2.0e-4
SAVE_EVERY = 50
REF_SAVE_EVERY = 50

# Direction convention for Poincare sections:
#  0 = both directions, +1 = upward tau crossing, -1 = downward tau crossing.
POINCARE_DIRECTIONS = [0, +1, -1]

OUT_ROOT = Path("results_successful_phase_poincare_mobius")

# Optional plotting limits. Keep None for automatic.
NUSS_YLIM = None              # example: (1.0, 3.2)
PHASE3D_TAU_LIM = None        # example: (-0.45, 0.45)
PHASE3D_PSI_LIM = None        # example: (-8.0, 8.0)
PHASE3D_NUT_LIM = None        # example: (1.0, 1.9)
PHASE3D_NUS_LIM = None
PHASE3D_SAL_LIM = None        # example: (-0.45, 0.45)

# Cosmetic / animation outputs. HTML is lightweight. GIF/MP4 require imageio and can be slower.
MAKE_INTERACTIVE_3D_HTML = True
MAKE_COSMIC_NEBULA_HTML = True
MAKE_COSMIC_GIF = False
MAKE_COSMIC_MP4 = False
MAKE_ELECTRON_GIF = False
MAKE_ELECTRON_MP4 = False


# ============================================================
# SMALL UTILS
# ============================================================

def fmt_rt(rt: float) -> str:
    return (f"{rt:g}").replace(".", "p")


def set_solver_case(rt: float, outdir: Path) -> None:
    """Patch the numerical-core module globals for this case."""
    bg.RT = float(rt)
    bg.RS = float(RS)
    bg.SIGMA = float(SIGMA)
    bg.KAPPA = float(KAPPA)
    bg.TAU_SECTION = float(TAU_SECTION)
    bg.PHASE_TMIN = float(PHASE_TMIN)
    bg.NUSS_TMIN = float(PHASE_TMIN)
    bg.NUSS_TMAX = float(T_TOTAL)
    bg.T_TOTAL = float(T_TOTAL)
    bg.REF_T_TOTAL = float(T_TOTAL)
    bg.DT = float(DT)
    bg.SAVE_EVERY = int(SAVE_EVERY)
    bg.REF_SAVE_EVERY = int(REF_SAVE_EVERY)
    bg.N_TRAJ = 1
    bg.PERT_MAIN = 0.0
    bg.PERT_SMALL = 0.0
    bg.USE_STEP_LEVEL_POINCARE_EVENTS = True
    bg.NUSS_YLIM = NUSS_YLIM
    bg.PHASE3D_TAU_LIM = PHASE3D_TAU_LIM
    bg.PHASE3D_PSI_LIM = PHASE3D_PSI_LIM
    bg.PHASE3D_NUT_LIM = PHASE3D_NUT_LIM
    bg.PHASE3D_NUS_LIM = PHASE3D_NUS_LIM
    bg.PHASE3D_SAL_LIM = PHASE3D_SAL_LIM

    bg.MAKE_INTERACTIVE_3D_HTML = MAKE_INTERACTIVE_3D_HTML
    bg.MAKE_COSMIC_NEBULA_HTML = MAKE_COSMIC_NEBULA_HTML
    bg.MAKE_COSMIC_GIF = MAKE_COSMIC_GIF
    bg.MAKE_COSMIC_MP4 = MAKE_COSMIC_MP4
    bg.MAKE_ARTISTIC_STYLE_ELECTRONS_GIF = MAKE_ELECTRON_GIF
    bg.MAKE_ARTISTIC_STYLE_ELECTRONS_MP4 = MAKE_ELECTRON_MP4

    # Keep outputs local to the case folder if someone calls bg.main().
    bg.OUT_PHASE = str(outdir / "phase_projection.png")
    bg.OUT_NU = str(outdir / "nusselts_time_history.png")
    bg.OUT_POINCARE_REF = str(outdir / "poincare_both.png")
    bg.OUT_POINCARE_REF_UP = str(outdir / "poincare_upward.png")
    bg.OUT_POINCARE_REF_DOWN = str(outdir / "poincare_downward.png")
    bg.OUT_POINCARE_EVENTS_CSV = str(outdir / "poincare_events_tau_m0p36.csv")
    bg.OUT_PHASE_3D_SAL = str(outdir / "phase_3d_tau11_psi11_s11.png")
    bg.OUT_PHASE_3D_NUT = str(outdir / "phase_3d_tau11_psi11_NuT.png")
    bg.OUT_PHASE_3D_SAL_HTML = str(outdir / "interactive_phase_3d_s11.html")
    bg.OUT_PHASE_3D_NUT_HTML = str(outdir / "interactive_phase_3d_NuT.html")
    bg.OUT_COSMIC_NEBULA_HTML = str(outdir / "cosmic_nebula_phase_3d.html")
    bg.OUT_COSMIC_GIF = str(outdir / "cosmic_nebula_phase_3d.gif")
    bg.OUT_COSMIC_MP4 = str(outdir / "cosmic_nebula_phase_3d.mp4")
    bg.OUT_ARTISTIC_STYLE_3D_SAL = str(outdir / "artistic_phase_3d_s11.png")
    bg.OUT_ARTISTIC_STYLE_ELECTRONS_GIF = str(outdir / "artistic_electrons_phase_3d.gif")
    bg.OUT_ARTISTIC_STYLE_ELECTRONS_MP4 = str(outdir / "artistic_electrons_phase_3d.mp4")


def save_npz_trajectory(tr: Dict[str, np.ndarray], events: Dict[str, np.ndarray], outfile: Path) -> None:
    np.savez_compressed(
        outfile,
        t=tr["t"], tau=tr["tau"], psi=tr["psi"], sal=tr["sal"], NuT=tr["NuT"], NuS=tr["NuS"],
        event_t=events["t"], event_psi=events["psi"], event_NuT=events["NuT"],
        event_NuS=events["NuS"], event_direction=events["direction"],
    )


def write_events_csv(events: Dict[str, np.ndarray], outfile: Path) -> None:
    with outfile.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "psi11", "NuT", "NuS", "direction"])
        for row in zip(events["t"], events["psi"], events["NuT"], events["NuS"], events["direction"]):
            w.writerow(row)


# ============================================================
# PLOTTING
# ============================================================

def normal_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=6, width=0.8, labelsize=12)


def centered_axes(ax):
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_color("0.55")
    ax.spines["bottom"].set_color("0.55")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=6, width=0.8, labelsize=12)


def plot_phase_projection(tr: Dict[str, np.ndarray], rt: float, outdir: Path) -> None:
    mask = tr["t"] >= PHASE_TMIN
    fig, ax = plt.subplots(figsize=(8.0, 8.0), facecolor="white")
    ax.plot(tr["tau"][mask], tr["psi"][mask], "k-", lw=0.35, alpha=0.85)
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-8.0, 8.0)
    ax.set_xticks(np.arange(-0.4, 0.41, 0.1))
    ax.set_yticks(np.arange(-8, 9, 2))
    centered_axes(ax)
    ax.text(0.50, 0.955, r"$\psi_{11}$", transform=ax.transAxes, ha="center", va="top", fontsize=24)
    ax.text(0.965, 0.515, r"$\tau_{11}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=24)
    ax.set_title("Phase projection", fontsize=24, pad=16)
    fig.text(0.20, 0.055, fr"$R_T={rt:g},\quad R_S={RS:g},\quad Pr={SIGMA:g},\quad \kappa={KAPPA:.5g}$", fontsize=16)
    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.96])
    fig.savefig(outdir / "phase_projection_tau11_psi11.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_phase_3d(tr: Dict[str, np.ndarray], rt: float, outdir: Path, zkey: str = "sal", elev: float = 18, azim: float = 35) -> None:
    mask = tr["t"] >= PHASE_TMIN
    zlabel = {"sal": r"$s_{11}$", "NuT": r"$Nu_T$", "NuS": r"$Nu_S$"}[zkey]
    fig = plt.figure(figsize=(8.5, 7.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tr["tau"][mask], tr["psi"][mask], tr[zkey][mask], color="black", lw=0.45, alpha=0.85)
    ax.set_xlabel(r"$\tau_{11}$", fontsize=13, labelpad=14)
    ax.set_ylabel(r"$\psi_{11}$", fontsize=13, labelpad=16)
    ax.set_zlabel(zlabel, fontsize=13, labelpad=18)
    if PHASE3D_TAU_LIM is not None:
        ax.set_xlim(*PHASE3D_TAU_LIM)
    if PHASE3D_PSI_LIM is not None:
        ax.set_ylim(*PHASE3D_PSI_LIM)
    if zkey == "sal" and PHASE3D_SAL_LIM is not None:
        ax.set_zlim(*PHASE3D_SAL_LIM)
    if zkey == "NuT" and PHASE3D_NUT_LIM is not None:
        ax.set_zlim(*PHASE3D_NUT_LIM)
    if zkey == "NuS" and PHASE3D_NUS_LIM is not None:
        ax.set_zlim(*PHASE3D_NUS_LIM)
    ax.set_title(fr"3D phase check, $R_T={rt:g}$", fontsize=18, pad=18)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.75))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.90)
    fig.savefig(outdir / f"phase_3d_tau11_psi11_{zkey}.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plot_poincare(psi_cross: np.ndarray, rt: float, outdir: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0), facecolor="white")
    if len(psi_cross) >= 2:
        x = psi_cross[:-1]
        y = psi_cross[1:]
        ax.plot(x, y, "k.", ms=2.2)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        dx = max(xmax - xmin, 1e-4)
        dy = max(ymax - ymin, 1e-4)
        pad = 0.08 * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    normal_axes(ax)
    ax.set_xlabel(r"$\psi_{11}^{(n)}$", fontsize=18, labelpad=10)
    ax.set_ylabel(r"$\psi_{11}^{(n+1)}$", fontsize=18, labelpad=12)
    ax.set_title(
        fr"Poincaré map at $\tau_{{11}}={TAU_SECTION:g}$" + "\n" +
        fr"RT={rt:g}, RS={RS:g}, {label}, crossings={len(psi_cross)}",
        fontsize=15,
    )
    plt.tight_layout()
    fig.savefig(outdir / f"poincare_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_section_fluxes(events: Dict[str, np.ndarray], rt: float, outdir: Path) -> None:
    psi = events["psi"]
    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor="white")
    ax.plot(psi, events["NuT"], "k.", ms=2.2, label=r"$Nu_T$")
    ax.plot(psi, events["NuS"], ".", color="black", alpha=0.45, ms=2.2, label=r"$Nu_S$")
    if NUSS_YLIM is not None:
        ax.set_ylim(*NUSS_YLIM)
    normal_axes(ax)
    ax.set_xlabel(r"$\psi_{11}$", fontsize=18)
    ax.set_ylabel(r"$Nu$", fontsize=18)
    ax.legend(frameon=False, fontsize=14)
    ax.set_title(fr"Section-based fluxes at $\tau_{{11}}={TAU_SECTION:g}$, $R_T={rt:g}$", fontsize=16)
    plt.tight_layout()
    fig.savefig(outdir / "section_based_fluxes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nusselt_time_history(tr: Dict[str, np.ndarray], rt: float, outdir: Path) -> None:
    mask = (tr["t"] >= PHASE_TMIN) & (tr["t"] <= T_TOTAL)
    tt = tr["t"][mask]
    nuT = tr["NuT"][mask]
    nuS = tr["NuS"][mask]
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white")
    ax.plot(tt, nuT, "k-", lw=0.9, label=r"$Nu_T$")
    ax.plot(tt, nuS, color="black", alpha=0.55, lw=0.75, label=r"$Nu_S$")
    ax.set_xlim(PHASE_TMIN, T_TOTAL)
    if NUSS_YLIM is not None:
        ax.set_ylim(*NUSS_YLIM)
    elif len(tt):
        ymin = min(float(np.min(nuT)), float(np.min(nuS)))
        ymax = max(float(np.max(nuT)), float(np.max(nuS)))
        dy = max(ymax-ymin, 1e-8)
        ax.set_ylim(ymin-0.08*dy, ymax+0.08*dy)
    normal_axes(ax)
    ax.set_xlabel(r"$t$", fontsize=18)
    ax.set_ylabel(r"$Nu$", fontsize=18)
    ax.legend(frameon=False, fontsize=14)
    ax.set_title(fr"Time history of boundary fluxes, $R_T={rt:g}$", fontsize=18)
    plt.tight_layout()
    fig.savefig(outdir / "nusselts_time_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PRACTICAL MOBIUS-BAND CHECKS
# ============================================================

def mobius_overlap_separation(tr: Dict[str, np.ndarray], outdir: Path, bins: int = 90, min_count: int = 8) -> Tuple[int, float]:
    """
    Check whether the 2D projection has over/under separation in s_11.

    A genuine folded/twisted ribbon should often show bins where nearly the
    same (tau_11, psi_11) location corresponds to noticeably different s_11.
    This is not a rigorous topology proof, but it is a very useful sanity check.
    """
    mask = tr["t"] >= PHASE_TMIN
    tau = tr["tau"][mask]
    psi = tr["psi"][mask]
    sal = tr["sal"][mask]

    H_count, xedges, yedges = np.histogram2d(tau, psi, bins=bins)
    sal_min = np.full_like(H_count, np.inf, dtype=float)
    sal_max = np.full_like(H_count, -np.inf, dtype=float)

    ix = np.clip(np.searchsorted(xedges, tau, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(yedges, psi, side="right") - 1, 0, bins - 1)

    for i, j, s in zip(ix, iy, sal):
        if s < sal_min[i, j]: sal_min[i, j] = s
        if s > sal_max[i, j]: sal_max[i, j] = s

    valid = H_count >= min_count
    separation = np.where(valid, sal_max - sal_min, np.nan)
    max_sep = float(np.nanmax(separation)) if np.any(valid) else float("nan")
    n_overlap = int(np.sum(valid & (separation > 0.05 * max_sep))) if np.isfinite(max_sep) and max_sep > 0 else 0

    # Save table of bins with largest separation.
    rows = []
    for i in range(bins):
        for j in range(bins):
            if valid[i, j] and np.isfinite(separation[i, j]):
                rows.append((separation[i, j], H_count[i, j], 0.5*(xedges[i]+xedges[i+1]), 0.5*(yedges[j]+yedges[j+1])))
    rows.sort(reverse=True)
    with (outdir / "mobius_overlap_separation_bins.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["salinity_range", "count", "tau11_bin_center", "psi11_bin_center"])
        w.writerows(rows[:500])

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    vals = separation[np.isfinite(separation)]
    if len(vals):
        ax.hist(vals, bins=40, color="black", alpha=0.85)
    normal_axes(ax)
    ax.set_xlabel(r"within-bin range of $s_{11}$", fontsize=14)
    ax.set_ylabel("number of projection bins", fontsize=14)
    ax.set_title("Projection over/under separation diagnostic", fontsize=16)
    plt.tight_layout()
    fig.savefig(outdir / "mobius_overlap_separation_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return n_overlap, max_sep


def mobius_local_normal_check(tr: Dict[str, np.ndarray], outdir: Path, stride: int = 12, window: int = 61) -> float:
    """
    A lightweight orientation-continuity diagnostic.

    We estimate local normals of the 3D curve cloud using PCA windows along time,
    then orient consecutive normals continuously. If a closed traversal reverses
    the normal direction, dot(n_end, n_start) tends negative. For chaotic bands
    this is only a diagnostic, not a theorem.
    """
    mask = tr["t"] >= PHASE_TMIN
    P = np.column_stack([tr["tau"][mask], tr["psi"][mask], tr["sal"][mask]])
    if len(P) < window + 5:
        return float("nan")
    P = P[::stride]
    half = window // 2
    normals = []
    centers = []
    for i in range(half, len(P) - half):
        Q = P[i-half:i+half+1]
        C = Q - Q.mean(axis=0)
        _, _, vh = np.linalg.svd(C, full_matrices=False)
        n = vh[-1]
        if normals and np.dot(normals[-1], n) < 0:
            n = -n
        normals.append(n)
        centers.append(P[i])
    normals = np.asarray(normals)
    centers = np.asarray(centers)
    dot_end_start = float(np.dot(normals[0], normals[-1])) if len(normals) else float("nan")

    dots = normals @ normals[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
    ax.plot(dots, "k-", lw=0.8)
    ax.axhline(0.0, color="0.5", lw=0.8)
    normal_axes(ax)
    ax.set_xlabel("ordered local-window index", fontsize=13)
    ax.set_ylabel(r"$n_i \cdot n_0$", fontsize=14)
    ax.set_title(fr"Local-normal continuity check, end/start dot = {dot_end_start:.3f}", fontsize=15)
    plt.tight_layout()
    fig.savefig(outdir / "mobius_local_normal_dot_check.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return dot_end_start


def run_mobius_checks(tr: Dict[str, np.ndarray], rt: float, outdir: Path) -> None:
    mobdir = outdir / "mobius_checks"
    mobdir.mkdir(exist_ok=True)

    plot_phase_3d(tr, rt, mobdir, zkey="sal", elev=18, azim=35)
    plot_phase_3d(tr, rt, mobdir, zkey="NuT", elev=24, azim=-58)

    n_overlap, max_sep = mobius_overlap_separation(tr, mobdir)
    end_dot = mobius_local_normal_check(tr, mobdir)

    with (mobdir / "mobius_diagnostic_report.txt").open("w") as f:
        f.write("Practical Mobius-band diagnostic report\n")
        f.write("========================================\n\n")
        f.write(f"RT = {rt:g}\nRS = {RS:g}\nPr = {SIGMA:g}\nkappa = {KAPPA:.8g}\n")
        f.write(f"tau section = {TAU_SECTION:g}\nphase tmin = {PHASE_TMIN:g}\n\n")
        f.write("Projection-overlap diagnostic:\n")
        f.write(f"  bins with clear over/under s11 separation = {n_overlap}\n")
        f.write(f"  maximum within-bin s11 separation = {max_sep:.8g}\n\n")
        f.write("Local-normal diagnostic:\n")
        f.write(f"  dot(n_end, n_start) = {end_dot:.8g}\n\n")
        f.write("Interpretation notes:\n")
        f.write("  - Large over/under separation supports the idea that the apparent 2D self-overlap\n")
        f.write("    is a projection of a 3D ribbon-like object.\n")
        f.write("  - A negative local-normal end/start dot is consistent with orientation reversal,\n")
        f.write("    but chaotic sampling makes this a diagnostic rather than a rigorous proof.\n")
        f.write("  - Use the saved 3D salinity figure together with the Poincare maps for the final call.\n")


# ============================================================
# CASE RUNNER
# ============================================================

def run_case(rt: float, ops: Dict[str, np.ndarray]) -> None:
    outdir = OUT_ROOT / f"RT_{fmt_rt(rt)}_RS_{fmt_rt(RS)}"
    outdir.mkdir(parents=True, exist_ok=True)
    set_solver_case(rt, outdir)

    print("\n" + "="*90)
    print(f"Running RT={rt:g}, RS={RS:g}, Pr={SIGMA:g}, kappa={KAPPA:.6g}")
    print(f"Output folder: {outdir}")
    print("="*90)

    y0 = bg.build_base_ic()
    tr, events = bg.integrate_reference_with_step_events(y0, ops)

    save_npz_trajectory(tr, events, outdir / "trajectory_and_poincare_events.npz")
    write_events_csv(events, outdir / "poincare_events_tau_m0p36.csv")

    plot_phase_projection(tr, rt, outdir)
    plot_phase_3d(tr, rt, outdir, zkey="sal", elev=18, azim=35)
    plot_phase_3d(tr, rt, outdir, zkey="NuT", elev=24, azim=-58)
    plot_nusselt_time_history(tr, rt, outdir)

    # Optional interactive/cosmetic outputs using the solver's visualization helpers.
    traj_list = [tr]
    if MAKE_INTERACTIVE_3D_HTML:
        try:
            bg.plot_phase_3d_interactive(traj_list, use="sal", tmin=PHASE_TMIN, outfile=str(outdir / "interactive_phase_3d_s11.html"))
            bg.plot_phase_3d_interactive(traj_list, use="NuT", tmin=PHASE_TMIN, outfile=str(outdir / "interactive_phase_3d_NuT.html"))
        except Exception as exc:
            print(f"Interactive 3D export skipped/failed: {exc}")
    if MAKE_COSMIC_NEBULA_HTML:
        try:
            bg.plot_phase_3d_cosmic_nebula_animation(traj_list, use="NuT", tmin=PHASE_TMIN, outfile=str(outdir / "cosmic_nebula_phase_3d_NuT.html"))
            bg.plot_phase_3d_cosmic_nebula_animation(traj_list, use="sal", tmin=PHASE_TMIN, outfile=str(outdir / "cosmic_nebula_phase_3d_s11.html"))
        except Exception as exc:
            print(f"Cosmic-nebula HTML export skipped/failed: {exc}")
    if MAKE_COSMIC_GIF or MAKE_COSMIC_MP4:
        try:
            bg.export_phase_3d_cosmic_gif_mp4(traj_list, use="NuT", tmin=PHASE_TMIN, gif_out=str(outdir / "cosmic_nebula_phase_3d.gif"), mp4_out=str(outdir / "cosmic_nebula_phase_3d.mp4"), export_gif=MAKE_COSMIC_GIF, export_mp4=MAKE_COSMIC_MP4)
        except Exception as exc:
            print(f"Cosmic-nebula GIF/MP4 export skipped/failed: {exc}")
    if MAKE_ELECTRON_GIF or MAKE_ELECTRON_MP4:
        try:
            bg.save_lorenz_style_3d_projected_electrons_animation(traj_list, use="sal", tmin=PHASE_TMIN, gif_out=str(outdir / "artistic_electrons_phase_3d.gif"), mp4_out=str(outdir / "artistic_electrons_phase_3d.mp4"), export_gif=MAKE_ELECTRON_GIF, export_mp4=MAKE_ELECTRON_MP4)
        except Exception as exc:
            print(f"Electron animation export skipped/failed: {exc}")

    psic = events["psi"]
    dirs = events["direction"]
    plot_poincare(psic, rt, outdir, "both")
    if np.any(dirs == +1):
        plot_poincare(psic[dirs == +1], rt, outdir, "upward")
    if np.any(dirs == -1):
        plot_poincare(psic[dirs == -1], rt, outdir, "downward")

    plot_section_fluxes(events, rt, outdir)
    run_mobius_checks(tr, rt, outdir)

    print(f"Done RT={rt:g}. Crossings: total={len(psic)}, up={np.sum(dirs==+1)}, down={np.sum(dirs==-1)}")


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    ops = bg.load_operators()

    # Warm up numba once.
    print("Compiling RK4 kernel once...")
    y_dummy = bg.build_base_ic()
    _ = bg.rk4_step_coeff(
        y_dummy, bg.DT, SIGMA, RT_CASES[0], RS, KAPPA,
        ops["lam_psi"], ops["lam_sc"], ops["Qpsi"], ops["Qtau"], ops["Qsal"],
        ops["Bpsi_sc"], ops["Fsc_psi"],
    )
    print("Compilation done.")

    for rt in RT_CASES:
        run_case(float(rt), ops)

    print("\nAll requested cases finished.")
    print(f"Results are in: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
