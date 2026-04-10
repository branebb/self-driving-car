"""
Pacejka Magic Formula Tire Model
=================================

The Pacejka Magic Formula is the industry-standard mathematical model for
tire force generation.  It was developed by Hans Pacejka at Delft University
(Netherlands) across a series of papers in the 1980s–90s and is used by
virtually every serious vehicle simulation — from F1 team lap-time simulators
to commercial vehicle-dynamics software (CarSim, Adams/Car, etc.).

The formula produces a characteristic S-curve that matches real measured tire
data remarkably well with only four parameters:

  - Linear region   (small slip): force grows nearly proportionally to slip
  - Peak region     (optimal slip): maximum force — the 'limit of grip'
  - Falloff region  (large slip): force drops as the tire slides

Reference:
  Pacejka, H.B. (2012) Tyre and Vehicle Dynamics, 3rd ed., Butterworth-Heinemann.
  Original formula: Pacejka & Bakker (1992) Vehicle System Dynamics, 21(sup1), 1–18.
"""

import numpy as np


def pacejka(x, B=10.0, C=1.9, D=1.0, E=0.97):
    """
    Pacejka Magic Formula (pure slip version).

        y = D · sin( C · arctan( B·x − E·(B·x − arctan(B·x)) ) )

    This is the 'pure' version — either pure longitudinal (slip ratio)
    or pure lateral (slip angle) force, not combined slip.

    Parameters
    ----------
    x : float or array-like
        Input slip variable.
        • For longitudinal force:  x = slip ratio  (dimensionless, typical range −1 to +1)
        • For lateral force:       x = slip angle  (radians)
    B : float
        Stiffness factor.  Controls the initial slope of the curve.
        Higher B → stiffer, more responsive tire (sharper onset of grip).
        F1 slick typical: B ≈ 10 – 14.
    C : float
        Shape factor.  Controls the 'width' of the peak and how quickly the
        curve transitions from linear → peak → falloff.
        Typical range: 1.0 – 2.5.
        C = 1.9  → lateral force shape (wider peak, moderate falloff).
        C = 1.65 → longitudinal force shape (slightly narrower peak).
    D : float
        Peak value — the maximum normalised force.
        In normalised form: D = 1.0 means μ_peak = 1.0 (one unit of normal force).
        For an F1 slick on dry tarmac: D ≈ 1.5 – 2.0.
    E : float
        Curvature / shape factor near and after the peak.
        E controls the sharpness of the falloff beyond peak slip.
        E = 0   → very early, sharp peak (Formula SAE-style stiff tire)
        E = 1   → moderate, symmetric S-curve (most road tires)
        E > 1   → allows force to fall below zero at very large slip
                  (not physically meaningful for most uses; keep E ≤ 1)
        Typical: 0.90 – 1.00 for road and race tires.

    Returns
    -------
    float or np.ndarray
        Normalised force (same sign as x).
        Multiply by normal load Fz (N) to get force in Newtons.

    Examples
    --------
    >>> # F1 slick at optimal braking slip
    >>> pacejka(-0.12, B=12, C=1.65, D=1.6, E=0.97)
    # → approximately −1.55  (i.e. μ ≈ 1.55 in the braking direction)

    >>> # Same tire at extreme wheelspin
    >>> pacejka(2.0, B=12, C=1.65, D=1.6, E=0.97)
    # → smaller positive value (degraded grip from sliding)
    """
    x  = np.asarray(x, dtype=float)
    Bx = B * x
    return D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))


def pacejka_peak(B, C, D, E, slip_range=(0.0, 1.5), n_points=20000):
    """
    Numerically locate the peak of the Pacejka curve.

    Parameters
    ----------
    B, C, D, E : float
        Pacejka parameters (same as pacejka()).
    slip_range : tuple (float, float)
        Search range for slip.  Default (0.0, 1.5) covers all
        physically meaningful positive slip values.
    n_points : int
        Resolution of the numerical search.

    Returns
    -------
    slip_at_peak : float
        The slip value (positive) at which peak force occurs.
    peak_force : float
        The peak normalised force (= D · sin(…) at its maximum).

    Notes
    -----
    The formula is odd-symmetric: pacejka(−x) = −pacejka(x), so the
    negative-slip peak is at (−slip_at_peak, −peak_force).
    """
    slips  = np.linspace(slip_range[0], slip_range[1], n_points)
    forces = pacejka(slips, B, C, D, E)
    idx    = int(np.argmax(forces))
    return float(slips[idx]), float(forces[idx])
