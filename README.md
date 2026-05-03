# Bubnov-Galerkin Double-Diffusive Convection: Phase, Poincaré, and Möbius Diagnostics

## Main scripts

### `dd_convection_galerkin_coeff_solver.py`

Single-case solver. Run this if you want the classic one-case workflow:

```bash
python dd_convection_galerkin_coeff_solver.py
```

It produces:

- paper-style phase projection `(tau_11, psi_11)`
- black scientific 3D phase portraits `(tau_11, psi_11, Nu_T)` and `(tau_11, psi_11, s_11)`
- Poincaré maps at `tau_11 = -0.36`
- upward/downward companion maps
- section-based Nusselt plots
- lightweight Möbius/twisted-ribbon diagnostics:
  - `mobius_check_overlap_separation.png`
  - `mobius_check_normal_dot.png`
  - `mobius_check_report.txt`

### `run_successful_phase_poincare_mobius.py`

Batch runner for the agreed successful cases:

```bash
python run_successful_phase_poincare_mobius.py
```

Default cases:

```python
RT_CASES = [9098.0, 9103.0, 9110.0]
```

Each case is saved in:

```text
results_successful_phase_poincare_mobius/RT_<value>_RS_8000/
```

## Important plotting options

At the top of `run_successful_phase_poincare_mobius.py`, you can set custom limits:

```python
NUSS_YLIM = None              # example: (1.0, 3.2)
PHASE3D_TAU_LIM = None        # example: (-0.45, 0.45)
PHASE3D_PSI_LIM = None        # example: (-8.0, 8.0)
PHASE3D_NUT_LIM = None        # example: (1.0, 1.9)
PHASE3D_NUS_LIM = None
PHASE3D_SAL_LIM = None        # example: (-0.45, 0.45)
```

The same options also exist near the top of `dd_convection_galerkin_coeff_solver.py` for direct one-case runs.

## Cosmetic and animation outputs

The batch runner includes the previous cosmetic/nebula visualization hooks:

```python
MAKE_INTERACTIVE_3D_HTML = True
MAKE_COSMIC_NEBULA_HTML = True
MAKE_COSMIC_GIF = False
MAKE_COSMIC_MP4 = False
MAKE_ELECTRON_GIF = False
MAKE_ELECTRON_MP4 = False
```

HTML output is usually safe and lightweight. GIF/MP4 rendering is slower and may require `imageio`, `pillow`, and possibly `ffmpeg`.

## Notes on the Poincaré map

The package uses step-level section crossing detection, not only downsampled saved trajectory points. This is important because the saved phase curve can look correct while the Poincaré map becomes sparse or distorted if crossings are extracted only from coarse saved output.

Direction convention:

```text
+1  upward crossing of tau_11 = -0.36
-1  downward crossing of tau_11 = -0.36
0   both directions / time-ordered event succession
```

## Install requirements

```bash
pip install -r requirements.txt
```

Core requirements are intentionally simple: NumPy, SciPy, Matplotlib, and Numba. Plotly/ImageIO/Pillow are included for optional interactive and animation outputs.
