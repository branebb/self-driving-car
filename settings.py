"""
Settings — F1 Simulation Tunable Parameters
============================================

Edit this file to customise the car physics, drivetrain, and aerodynamics.
All values are in SI units unless noted otherwise.

Three aerodynamic presets are included:
  AERO_SILVERSTONE  ← default (medium wing)
  AERO_MONZA            (low drag, high speed)
  AERO_MONACO           (maximum downforce, slow circuit)

Usage
-----
  from settings import *
  # or
  from settings import AERO_SILVERSTONE, CAR_MASS, GEAR_RATIOS
"""

# ── Vehicle geometry ──────────────────────────────────────────────────────────

CAR_MASS        = 798.0   # kg    total mass (car + driver, 2024 F1 minimum)
WHEELBASE       = 3.60    # m     CG_TO_FRONT + CG_TO_REAR
CG_TO_FRONT     = 1.94    # m     CG to front axle  (46 % weight on front)
CG_TO_REAR      = 1.66    # m     CG to rear axle   (54 % weight on rear)
YAW_INERTIA     = 800.0   # kg·m² yaw moment of inertia (real F1 estimate: 700-1000 kg·m²)
CG_HEIGHT       = 0.30    # m     CG height above ground (F1 is extremely low)
TRACK_WIDTH_F   = 1.60    # m     front track width (contact-patch centre–centre)
TRACK_WIDTH_R   = 1.60    # m     rear  track width

# ── Tyre geometry (for rendering) ─────────────────────────────────────────────

WHEEL_RADIUS    = 0.330   # m     ~13-inch rim + tyre (~660 mm diameter)
TIRE_LENGTH     = 0.705   # m     tire visual length for top-down rendering (~outer wheel diameter); NOT the contact patch
TIRE_WIDTH_M    = 0.290   # m     contact-patch width

# ── Drivetrain ────────────────────────────────────────────────────────────────

RPM_IDLE        = 800
RPM_REDLINE     = 15000

GEAR_RATIOS = {
    'R': -4.50,   # Reverse
    'N':  0.00,   # Neutral
    1:    7.00,   # shifts up at  ~73 km/h  (real F1: ~70-80 km/h)
    2:    5.00,   # shifts up at ~103 km/h
    3:    3.75,   # shifts up at ~137 km/h
    4:    2.95,   # shifts up at ~174 km/h
    5:    2.40,   # shifts up at ~214 km/h
    6:    1.98,   # shifts up at ~259 km/h
    7:    1.71,   # shifts up at ~300 km/h
    8:    1.52,   # top speed   ~370 km/h at 15 000 RPM
}

FINAL_DRIVE_RATIO = 3.27  # differential ratio (always multiplied on top of gear ratio)

# Engine torque curve  — Gaussian bell:
#   torque(rpm) = TORQUE_BASE + TORQUE_PEAK * exp(-TORQUE_WIDTH * (rpm - TORQUE_PEAK_RPM)^2)
TORQUE_BASE     = 50      # Nm   baseline torque at idle
TORQUE_PEAK     = 250     # Nm   extra torque at the power peak above baseline
TORQUE_WIDTH    = 1.5e-7  # RPM^-2  peak narrowness (large value = peaky F1 character)
TORQUE_PEAK_RPM = 11000   # RPM at peak torque

# Effective rear-wheel rotational inertia (wheel assembly + reflected drivetrain)
I_WHEEL_EFF     = 1.5     # kg*m^2 per driven wheel

# Rolling resistance (constant, opposes forward motion)
ROLLING_RESIST  = 0.015   # dimensionless (typical: 0.010-0.020 for slick on tarmac)

# ── Pacejka tyre parameters (F1 slick, dry tarmac) ───────────────────────────
#
#   The Magic Formula for a single axis:
#     F = Fz * mu(Fz) * sin(C * arctan(B*x - E*(B*x - arctan(B*x))))
#
#   where x is the slip input (kappa for longitudinal, tan(alpha) for lateral),
#   and the B, C, D, E parameters control the shape of the force curve.
#
#   B = stiffness factor   - steeper initial rise -> higher B
#   C = shape factor       - controls peak sharpness (C < 2 gives one peak)
#   D = peak friction      - set dynamically via load sensitivity (see below)
#   E = curvature factor   - sharpens/broadens the peak region (E < 1 typical)

B_LONGITUDINAL  = 12.0    # stiffer build-up than lateral (tyre construction)
C_LONGITUDINAL  = 1.65
E_LONGITUDINAL  = 0.97

B_LATERAL       = 10.0
C_LATERAL       = 1.90
E_LATERAL       = 0.97

# ── Load sensitivity ──────────────────────────────────────────────────────────
#
#   Peak friction coefficient as a function of normal load:
#     mu(Fz) = D1 - D2 * Fz
#
#   Rubber friction is sublinear in load: doubling the load less than doubles
#   the lateral force.  This means weight transfer always reduces total axle grip.
#
#   Calibrated so that:
#     mu = 1.60  at  Fz = 1800 N  (lightly loaded tyre)
#     mu = 1.35  at  Fz = 3600 N  (heavily loaded tyre)

D1 = 1.85       # intercept  (dimensionless)
D2 = 1.39e-4    # slope      (N^-1)

# ── Braking ───────────────────────────────────────────────────────────────────

KAPPA_PEAK_BRAKE = 0.15   # peak longitudinal slip ratio under ABS braking
BRAKE_BIAS_F     = 0.60   # fraction of total brake force applied to front axle

# ── Aerodynamic presets ───────────────────────────────────────────────────────
#
#   F_downforce = C_DF * Vx^2     (N - adds to tyre normal loads, boosting grip)
#   F_drag      = C_DR * Vx^2     (N - opposes forward motion)
#   AERO_BALANCE_F: fraction of downforce carried by the front axle
#
#   At 60 m/s (216 km/h) with Silverstone setup:
#     Downforce ~= 5400 N  (~69 % of car weight added as extra grip)
#     Drag      ~= 2304 N  (C_DR=0.64 × 3600)

AERO_SILVERSTONE = dict(   # Medium wing - general-purpose default
    C_DF=1.50,
    C_DR=0.64,
    AERO_BALANCE_F=0.37,
)

AERO_MONZA = dict(         # Low drag - fast straights, minimal wing
    C_DF=0.70,
    C_DR=0.32,
    AERO_BALANCE_F=0.37,
)

AERO_MONACO = dict(        # Maximum downforce - slow, twisty, bumpy
    C_DF=2.50,
    C_DR=1.00,
    AERO_BALANCE_F=0.37,
)

# Default aero (used by Full2DModel when no override is passed)
DEFAULT_AERO = AERO_SILVERSTONE

# ── Simulation defaults ───────────────────────────────────────────────────────

DT_DEFAULT  = 0.005   # s    recommended integration timestep (5 ms = 200 Hz)
AUTO_SHIFT  = True    # bool automatic gear changes (up at 90% redline, down at 40%)

# ── RL action space reference ─────────────────────────────────────────────────
#
#   For a gym-compatible environment the action vector is:
#     [throttle in [0,1],  brake in [0,1],  delta in [-MAX_STEER_RAD, +MAX_STEER_RAD]]
#
#   Gear control is not needed - AUTO_SHIFT handles shifting automatically.
#   Traction control is not modelled - wheelspin is physics-driven.

MAX_STEER_RAD = 0.35   # rad   maximum front steer angle (~20 deg, F1 typical lock)
