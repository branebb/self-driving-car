"""
Combined Slip Tire Model
========================

When a tire simultaneously has longitudinal slip (κ ≠ 0) AND lateral slip (α ≠ 0),
the two force components compete for the same friction resource at the contact patch.
You cannot have maximum braking force AND maximum cornering force at the same time —
this is the *friction circle* (or friction ellipse) constraint.

This module implements the **slip-vector decomposition** method:

  1. Compute combined slip magnitude: σ = sqrt(κ² + tan²α)
  2. Decompose: each direction contributes a fraction of σ
  3. Each direction's Pacejka is evaluated at σ, not at its individual slip alone
  4. Force vector = (slip direction unit vector) × (Pacejka(σ) × Fz)

Limiting cases (verified analytically):
  α = 0  →  Fx = Pacejka(κ,  lon) × Fz,   Fy = 0          ✓
  κ = 0  →  Fx = 0,                         Fy = Pacejka(tan(α), lat) × Fz  ✓

Reference:
  Pacejka H.B. (2012) Tyre and Vehicle Dynamics, 3rd ed., §4.3.
  Simplified combined-slip formulation after Bakker, Nyborg & Pacejka (1987)
  Vehicle System Dynamics 15(sup1), 1–15.
"""

import numpy as np
from pacejka import pacejka


def combined_slip_forces(
    kappa, alpha, Fz,
    B_lon=12.0, C_lon=1.65, D_lon=1.6, E_lon=0.97,
    B_lat=10.0, C_lat=1.90, D_lat=1.6, E_lat=0.97,
):
    """
    Combined slip tire forces via slip-vector decomposition.

    Parameters
    ----------
    kappa : float or array-like
        Longitudinal slip ratio (dimensionless).
        Positive = traction (driven wheel spins faster than vehicle).
        Negative = braking (wheel spins slower than vehicle).
    alpha : float or array-like
        Lateral slip angle (radians).  Positive = slip toward left (ISO).
    Fz : float
        Normal load on this tire / axle (N).
    B_lon, C_lon, D_lon, E_lon : float
        Pacejka parameters for the longitudinal direction.
    B_lat, C_lat, D_lat, E_lat : float
        Pacejka parameters for the lateral direction.

    Returns
    -------
    Fx : float or np.ndarray
        Longitudinal force (N).  Positive = thrust, negative = braking.
    Fy : float or np.ndarray
        Lateral force (N).  Positive = left (ISO).

    Notes
    -----
    The lateral slip component is expressed as tan(α) (not α) so that the
    combined slip vector has consistent physical units and the formula reduces
    correctly to the pure lateral case in the limit κ → 0.
    For small angles tan(α) ≈ α, so the difference is negligible for α < ~15°.

    The total force magnitude is bounded by:
        sqrt(Fx² + Fy²) ≤ max(D_lon, D_lat) × Fz
    This is the friction circle (or ellipse when D_lon ≠ D_lat).
    """
    kappa     = np.asarray(kappa, dtype=float)
    alpha     = np.asarray(alpha, dtype=float)
    tan_alpha = np.tan(alpha)

    # Combined slip magnitude — the 'effective total slip' seen by the contact patch
    sigma = np.sqrt(kappa ** 2 + tan_alpha ** 2)

    # Safe denominator (no force when there is no slip)
    tiny       = sigma < 1e-9
    safe_sigma = np.where(tiny, 1.0, sigma)

    # Each direction gets its share of the total force, evaluated at sigma
    Fx = np.where(
        tiny, 0.0,
        (kappa     / safe_sigma) * pacejka(safe_sigma, B_lon, C_lon, D_lon, E_lon) * Fz
    )
    Fy = np.where(
        tiny, 0.0,
        (tan_alpha / safe_sigma) * pacejka(safe_sigma, B_lat, C_lat, D_lat, E_lat) * Fz
    )

    # Return scalars when scalar inputs were given
    if Fx.ndim == 0:
        return float(Fx), float(Fy)
    return Fx, Fy
