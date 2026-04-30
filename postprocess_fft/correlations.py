"""Velocity correlation tensor, f(r), g(r), and derived length scales."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from .spectra import _extract_axis_line_from_local_volume
from .spectra import _shell_bin_geometry
from .spectra import _reduce_shell_histogram
from .transform import backward_field


# ------------------------------------------------------------------ #
#  Part A: Shell-binned spectrum tensor Phi_ij(k)
# ------------------------------------------------------------------ #

def compute_spectrum_tensor_diagonal(vx_k, vy_k, vz_k, shape, box, comm):
    """Shell-bin the diagonal components of the velocity spectrum tensor.

    Computes Phi_ii(|k|) = sum_{|k| in shell} |u_hat_i(k)/N|^2
    for i = 1, 2, 3.  No 0.5 prefactor is included; the raw
    spectrum-tensor definition is returned (energy spectra are
    E_ii = 0.5 * Phi_ii).

    Returns (on rank 0; None components on other ranks)
    -------
    k_bin_centers, Phi11, Phi22, Phi33
    """
    norm = float(np.prod(shape))
    phi11_density = np.abs(vx_k / norm) ** 2
    phi22_density = np.abs(vy_k / norm) ** 2
    phi33_density = np.abs(vz_k / norm) ** 2

    k_magnitude, k_bin_edges, k_bin_centers = _shell_bin_geometry(shape, box)
    Phi11 = _reduce_shell_histogram(k_magnitude, k_bin_edges, phi11_density, comm)
    Phi22 = _reduce_shell_histogram(k_magnitude, k_bin_edges, phi22_density, comm)
    Phi33 = _reduce_shell_histogram(k_magnitude, k_bin_edges, phi33_density, comm)
    return k_bin_centers, Phi11, Phi22, Phi33


def compute_spectrum_tensor_offdiagonal(vx_k, vy_k, vz_k, shape, box, comm):
    """Shell-bin the 3 unique off-diagonal components of the spectrum tensor.

    Returns (on rank 0; None components on other ranks)
    -------
    k_bin_centers, Re_Phi12, Re_Phi13, Re_Phi23,
                   Abs_Phi12, Abs_Phi13, Abs_Phi23
    """
    norm = float(np.prod(shape))
    vx_n = vx_k / norm
    vy_n = vy_k / norm
    vz_n = vz_k / norm

    phi12 = vx_n * np.conj(vy_n)
    phi13 = vx_n * np.conj(vz_n)
    phi23 = vy_n * np.conj(vz_n)

    k_magnitude, k_bin_edges, k_bin_centers = _shell_bin_geometry(shape, box)
    Re_Phi12 = _reduce_shell_histogram(k_magnitude, k_bin_edges, np.real(phi12), comm)
    Re_Phi13 = _reduce_shell_histogram(k_magnitude, k_bin_edges, np.real(phi13), comm)
    Re_Phi23 = _reduce_shell_histogram(k_magnitude, k_bin_edges, np.real(phi23), comm)
    Abs_Phi12 = _reduce_shell_histogram(k_magnitude, k_bin_edges, np.abs(phi12), comm)
    Abs_Phi13 = _reduce_shell_histogram(k_magnitude, k_bin_edges, np.abs(phi13), comm)
    Abs_Phi23 = _reduce_shell_histogram(k_magnitude, k_bin_edges, np.abs(phi23), comm)
    return k_bin_centers, Re_Phi12, Re_Phi13, Re_Phi23, Abs_Phi12, Abs_Phi13, Abs_Phi23


# ------------------------------------------------------------------ #
#  Part B: Two-point correlations R_ii(r) and f(r), g(r)
# ------------------------------------------------------------------ #

def compute_diagonal_correlation_axes(
    plan, vx_k, vy_k, vz_k, shape, local_shape, box, dx, dy, dz, comm,
):
    """Compute R_11, R_22, R_33 along the three principal axes.

    For each diagonal component ii:
      1. Phi_ii(k) = u_hat_i * conj(u_hat_i)  (pointwise, local)
      2. R_ii(r) = backward_FFT(Phi_ii) / N    (distributed backward FFT)
      3. Extract axis lines R_ii(r*e_b, 0, 0) for b = x, y, z
         using _extract_axis_line_from_local_volume with allreduce

    Uses the same HeFFTe plan as all other backward FFTs.

    Normalization: HeFFTe forward uses scale.none, backward uses scale.full
    (divides by N).  Phi_ii = |u_hat_i|^2 ~ N^2 * <u_i^2>, so
    backward(Phi_ii, full) ~ N * <u_i^2>.  Dividing by global_points
    gives the correct R_ii(r) = <u_i(x) u_i(x+r)>.
    """
    global_points = float(np.prod(shape))
    nmax = min(shape) // 2

    spacing = float((dx + dy + dz) / 3.0)
    r = np.arange(nmax + 1, dtype=np.float64) * spacing

    result = {"r": r}

    component_labels = ("vx", "vy", "vz")
    component_modes = (vx_k, vy_k, vz_k)
    diagonal_labels = ("R11", "R22", "R33")
    axis_labels = ("x", "y", "z")

    for diag_idx, (mode, diag_label) in enumerate(zip(component_modes, diagonal_labels)):
        phi_ii_k = mode * np.conj(mode)
        R_ii_local = backward_field(plan, phi_ii_k, local_shape)

        for axis in range(3):
            line = _extract_axis_line_from_local_volume(
                R_ii_local, shape, box, axis, comm,
            )
            line = line[:nmax + 1] / global_points
            result[f"{diag_label}_{axis_labels[axis]}"] = line

    return result


def extract_f_g(correlation_axes):
    """Extract normalized f(r) and g(r) from R_ii axis lines.

    f(r): longitudinal autocorrelation
        f_x(r) = R_11(r,0,0),  f_y(r) = R_22(0,r,0),  f_z(r) = R_33(0,0,r)
        f(r)   = (f_x + f_y + f_z) / 3,  normalized by f(0)

    g(r): transverse autocorrelation
        g_x(r) = 0.5*(R_22(r,0,0) + R_33(r,0,0))
        g_y(r) = 0.5*(R_11(0,r,0) + R_33(0,r,0))
        g_z(r) = 0.5*(R_11(0,0,r) + R_22(0,0,r))
        g(r)   = (g_x + g_y + g_z) / 3,  normalized by g(0)
    """
    r = correlation_axes["r"]

    f_x = correlation_axes["R11_x"]
    f_y = correlation_axes["R22_y"]
    f_z = correlation_axes["R33_z"]

    g_x = 0.5 * (correlation_axes["R22_x"] + correlation_axes["R33_x"])
    g_y = 0.5 * (correlation_axes["R11_y"] + correlation_axes["R33_y"])
    g_z = 0.5 * (correlation_axes["R11_z"] + correlation_axes["R22_z"])

    f_avg = (f_x + f_y + f_z) / 3.0
    g_avg = (g_x + g_y + g_z) / 3.0

    f_norm = f_avg / (f_avg[0] if np.abs(f_avg[0]) > 0 else 1.0)
    g_norm = g_avg / (g_avg[0] if np.abs(g_avg[0]) > 0 else 1.0)

    def _safe_normalize(curve):
        c0 = curve[0]
        if np.abs(c0) > 0.0:
            return curve / c0
        return np.full_like(curve, np.nan, dtype=np.float64)

    g_x_norm = _safe_normalize(g_x)
    g_y_norm = _safe_normalize(g_y)
    g_z_norm = _safe_normalize(g_z)

    return {
        "r": r,
        "f": f_norm,
        "g": g_norm,
        "f_x": f_x,
        "f_y": f_y,
        "f_z": f_z,
        "f_x_norm": _safe_normalize(f_x),
        "f_y_norm": _safe_normalize(f_y),
        "f_z_norm": _safe_normalize(f_z),
        "g_x": g_x,
        "g_y": g_y,
        "g_z": g_z,
        "g_x_norm": g_x_norm,
        "g_y_norm": g_y_norm,
        "g_z_norm": g_z_norm,
        "f_raw_avg": f_avg,
        "g_raw_avg": g_avg,
    }


# ------------------------------------------------------------------ #
#  Part C: Derived length scales
# ------------------------------------------------------------------ #

def _trapz_integral(y, x):
    """Compute a trapezoid-rule integral without depending on NumPy helpers."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.size < 2:
        return 0.0
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return float(np.sum(dx * avg, dtype=np.float64))

def second_derivative_at_origin(r, y):
    """Estimate y''(0) near the left boundary.

    Uses 5-point forward stencil O(h^4) when available:
      y''(0) = (35*y0 - 104*y1 + 114*y2 - 56*y3 + 11*y4) / (12*h^2)
    Falls back to 4-point forward stencil O(h^2):
      y''(0) = (2*y0 - 5*y1 + 4*y2 - y3) / h^2
    """
    if len(r) < 4 or len(y) < 4:
        raise ValueError("Need at least 4 points to estimate second derivative at r=0.")
    h = r[1] - r[0]
    if h <= 0.0:
        raise ValueError("Non-positive spacing in r.")
    ncheck = 5 if len(r) >= 5 else 4
    if not np.allclose(np.diff(r[:ncheck]), h, rtol=1e-10, atol=1e-14):
        raise ValueError("Non-uniform r spacing near origin.")

    if len(y) >= 5:
        return (
            35.0 * y[0] - 104.0 * y[1] + 114.0 * y[2] - 56.0 * y[3] + 11.0 * y[4]
        ) / (12.0 * h * h)

    return (2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]) / (h * h)


def compute_taylor_microscales(fg, eps=1e-14):
    """Compute Taylor microscales from f(r) and g(r).

    lambda_f = sqrt(-f(0) / f''(0))
    lambda_g = sqrt(-g(0) / g''(0))

    Also computes per-component longitudinal and transverse microscales.
    """
    r = fg["r"]

    # Averaged normalized curves
    lam_f, lam_g, d2f0, d2g0 = _taylor_microscales_from_curves(
        r, fg["f"], fg["g"],
    )

    # Per-component longitudinal: from f_x, f_y, f_z (unnormalized)
    f_comp, f_comp_avg = _component_microscales(fg, "f", eps)

    # Per-component transverse: from g_x, g_y, g_z (unnormalized)
    g_comp, g_comp_avg = _component_microscales(fg, "g", eps)

    return {
        "lambda_f": lam_f,
        "lambda_g": lam_g,
        "d2f0": d2f0,
        "d2g0": d2g0,
        "f_components": f_comp,
        "f_component_avg": f_comp_avg,
        "g_components": g_comp,
        "g_component_avg": g_comp_avg,
    }


def _taylor_microscales_from_curves(r, f_curve, g_curve):
    """Compute lambda_f, lambda_g from (possibly normalized) curves."""
    d2f0 = second_derivative_at_origin(r, f_curve)
    d2g0 = second_derivative_at_origin(r, g_curve)
    f0 = f_curve[0]
    g0 = g_curve[0]
    lam_f = np.sqrt(-f0 / d2f0) if d2f0 < 0.0 else np.nan
    lam_g = np.sqrt(-g0 / d2g0) if d2g0 < 0.0 else np.nan
    return lam_f, lam_g, d2f0, d2g0


def _component_microscales(fg, prefix, eps=1e-14):
    """Compute per-component Taylor microscales for f or g curves."""
    r = fg["r"]
    result = {}
    for tag in ("x", "y", "z"):
        curve = fg[f"{prefix}_{tag}"]
        c0 = curve[0]
        if np.abs(c0) <= eps:
            result[tag] = {"lambda": 0.0, "d2": 0.0, "c0": c0, "status": "zero_component"}
            continue
        d2 = second_derivative_at_origin(r, curve)
        lam = np.sqrt(-c0 / d2) if d2 < 0.0 else np.nan
        status = "ok" if np.isfinite(lam) else "invalid_curvature"
        result[tag] = {"lambda": lam, "d2": d2, "c0": c0, "status": status}

    lam_avg = (result["x"]["lambda"] + result["y"]["lambda"] + result["z"]["lambda"]) / 3.0
    return result, lam_avg


def compute_integral_length_scales(fg, eps=1e-14):
    """Compute integral length scales from normalized correlation curves.

    L_f = integral f_norm(r) dr
    L_g = integral g_norm(r) dr

    Also computes per-component longitudinal and transverse integral scales.
    """
    r = fg["r"]
    L_f = _trapz_integral(fg["f"], r)
    L_g = _trapz_integral(fg["g"], r)

    f_comp = {}
    g_comp = {}
    for tag in ("x", "y", "z"):
        f_c0 = fg[f"f_{tag}"][0]
        if np.abs(f_c0) <= eps:
            f_comp[tag] = {"L": 0.0, "status": "zero_component"}
        else:
            f_comp[tag] = {"L": _trapz_integral(fg[f"f_{tag}_norm"], r), "status": "ok"}

        g_c0 = fg[f"g_{tag}"][0]
        if np.abs(g_c0) <= eps:
            g_comp[tag] = {"L": 0.0, "status": "zero_component"}
        else:
            g_comp[tag] = {"L": _trapz_integral(fg[f"g_{tag}_norm"], r), "status": "ok"}

    L_f_comp_avg = (f_comp["x"]["L"] + f_comp["y"]["L"] + f_comp["z"]["L"]) / 3.0
    L_g_comp_avg = (g_comp["x"]["L"] + g_comp["y"]["L"] + g_comp["z"]["L"]) / 3.0

    return {
        "L_f": L_f,
        "L_g": L_g,
        "f_components": f_comp,
        "f_component_avg": L_f_comp_avg,
        "g_components": g_comp,
        "g_component_avg": L_g_comp_avg,
    }


def compute_longitudinal_integral_scale_from_spectrum(k_phy, e_phy_density, u1_variance):
    """Compute isotropic longitudinal integral scale from the 3D energy spectrum.

    For homogeneous isotropic turbulence,

      L_11 = (pi / (2 <u_1'^2>)) * integral_0^inf E(k) / k dk

    where E(k) is the physical 3D energy spectrum density and <u_1'^2> is the
    streamwise fluctuation variance. In anisotropic flows this should be treated
    as an additional isotropic-style summary, not a primary definition.
    """
    k_phy = np.asarray(k_phy, dtype=np.float64)
    e_phy_density = np.asarray(e_phy_density, dtype=np.float64)
    u1_variance = float(u1_variance)

    if k_phy.shape != e_phy_density.shape:
        raise ValueError("k_phy and e_phy_density must have the same shape.")
    if u1_variance <= 0.0:
        return np.nan

    mask = np.isfinite(k_phy) & np.isfinite(e_phy_density) & (k_phy > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.nan

    integral_value = _trapz_integral(e_phy_density[mask] / k_phy[mask], k_phy[mask])
    return float((np.pi / (2.0 * u1_variance)) * integral_value)
