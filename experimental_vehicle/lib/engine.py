"""
F1 Engine, Drivetrain, and Vehicle Constants
============================================

Extracted from experimental_vehicle/testing.ipynb.

Contains:
  - Vehicle and drivetrain parameters
  - torque_curve()     - Gaussian bell-curve engine torque model
  - get_wheel_force()  - full drivetrain calculation (engine → contact patch)
  - F1Engine           - 1D longitudinal physics simulation
"""

import numpy as np


# ── Drivetrain parameters ──────────────────────────────────────────────────────

# F1 gear ratios (realistic approximation).
# Each value = how many times the engine spins per one wheel rotation.
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

FINAL_DRIVE_RATIO = 3.27   # differential ratio (always applied on top of gear ratio)
WHEEL_RADIUS      = 0.330  # m  (F1 wheel ~13-inch rim + tire, ~660 mm diameter)
CAR_MASS          = 798    # kg (F1 car minimum with driver, 2024 rules)

RPM_IDLE    = 800
RPM_REDLINE = 15000

# ── Aerodynamic / resistance constants ────────────────────────────────────────

DRAG_COEFFICIENT = 0.7    # Cd - F1 has high drag due to front and rear wings
FRONTAL_AREA     = 1.5    # m² - frontal cross-section
AIR_DENSITY      = 1.225  # kg/m³ - sea level, 15 °C

# Pre-computed: 0.5 * Cd * A * rho   (used in drag force = AERO_DRAG_K * v²)
AERO_DRAG_K = 0.5 * DRAG_COEFFICIENT * FRONTAL_AREA * AIR_DENSITY

ROLLING_RESIST  = 0.015    # rolling resistance coefficient (dimensionless)
MAX_BRAKE_FORCE = 25000    # N - F1 carbon brakes are incredibly powerful


# ── Engine model ──────────────────────────────────────────────────────────────

def torque_curve(rpm, a=50, b=250, c=0.00000015, d=11000):
    """
    Engine torque (Nm) at a given RPM via a Gaussian bell-curve.

    Formula:  torque(rpm) = a + b * exp(-c * (rpm - d)²)

    Parameters
    ----------
    rpm : float
        Current engine RPM.
    a : float
        Baseline torque at idle (Nm).  F1 default: 50 Nm.
    b : float
        Extra torque above baseline at the peak (Nm).  F1 default: 250 Nm.
    c : float
        Peak narrowness.
        Large c → narrow, peaky power band (F1 character).
        Small c → wide, flat power band (diesel/truck character).
        F1 default: 1.5e-7 (very narrow peak).
    d : float
        RPM at peak torque.  F1 default: 11 000 RPM.

    Returns
    -------
    float
        Torque in Newton-metres (Nm).

    Notes
    -----
    With F1 defaults (c=1.5e-7, peak at 11 000 RPM) the curve produces
    essentially zero useful torque below ~8 000 RPM.  Demos that start
    from rest must either use a clutch-drop model (pre-spin the wheel to
    engine-coupled ω before releasing) or begin at a high-speed condition
    where RPM is already near peak.
    """
    return a + b * np.exp(-c * (rpm - d) ** 2)


# ── Drivetrain calculation ─────────────────────────────────────────────────────

def get_wheel_force(engine_rpm, gear, throttle=1.0):
    """
    Full drivetrain calculation: Engine RPM + gear → force at contact patch (N).

    Steps:
    1. Torque from RPM via torque_curve()
    2. × gear ratio
    3. × final drive ratio
    4. ÷ wheel radius → linear force at contact patch
    5. × throttle (0 – 1)

    Parameters
    ----------
    engine_rpm : float
        Current engine RPM.
    gear : int or str
        Current gear (1–8, 'N' for neutral, 'R' for reverse).
    throttle : float
        Throttle position 0.0 (closed) to 1.0 (wide open).

    Returns
    -------
    float
        Force in Newtons at the contact patch.
        Positive = forward thrust.  Negative = reverse.
    """
    if gear == 'N':
        return 0.0

    gear_ratio    = GEAR_RATIOS[gear]
    engine_torque = torque_curve(engine_rpm)
    wheel_torque  = engine_torque * abs(gear_ratio) * FINAL_DRIVE_RATIO
    wheel_force   = wheel_torque / WHEEL_RADIUS

    direction = -1 if gear == 'R' else 1
    return direction * wheel_force * throttle


# ── 1-D longitudinal simulation ───────────────────────────────────────────────

class F1Engine:
    """
    Simple 1-D F1 engine + drivetrain simulation.

    No lateral forces — just pure forward/backward physics.

    State
    -----
    rpm      : current engine RPM
    gear     : current gear (1–8)
    velocity : car speed (m/s)
    position : car position (m)
    time     : elapsed simulation time (s)
    history  : dict of per-step recorded values for plotting
    """

    def __init__(self):
        self.rpm      = RPM_IDLE
        self.gear     = 1
        self.velocity = 0.0
        self.position = 0.0
        self.time     = 0.0

        self.history = {
            'time':          [],
            'velocity_ms':   [],
            'velocity_kmh':  [],
            'rpm':           [],
            'gear':          [],
            'position':      [],
            'engine_force':  [],
            'drag_force':    [],
            'net_force':     [],
            'acceleration':  [],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rpm_from_velocity(self, velocity, gear):
        """
        Back-calculate the engine RPM that matches current wheel speed and gear.

        wheel_ω  = velocity / R_wheel           (rad/s)
        engine_ω = wheel_ω × gear_ratio × FDR   (rad/s)
        RPM      = engine_ω × 60 / (2π)
        """
        if gear == 'N' or velocity <= 0:
            return RPM_IDLE

        wheel_omega  = velocity / WHEEL_RADIUS
        engine_omega = wheel_omega * GEAR_RATIOS[gear] * FINAL_DRIVE_RATIO
        rpm = engine_omega * (60 / (2 * np.pi))
        return float(np.clip(rpm, RPM_IDLE, RPM_REDLINE))

    def _auto_shift(self):
        """
        Simple automatic gearbox.
        Shift up at 90 % of redline; shift down below 40 % of redline.
        """
        if self.rpm > RPM_REDLINE * 0.90 and self.gear < 8:
            self.gear += 1
        elif self.rpm < RPM_REDLINE * 0.40 and self.gear > 1:
            self.gear -= 1

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self, throttle, brake, dt=0.01, auto_shift=True):
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        throttle : float   0 – 1, fraction of wide-open throttle
        brake    : float   0 – 1, fraction of maximum brake force
        dt       : float   timestep in seconds (default 0.01 s = 100 Hz)
        auto_shift : bool  whether to allow automatic gear changes
        """
        # 1. Sync RPM to current wheel speed
        self.rpm = self._rpm_from_velocity(self.velocity, self.gear)

        # 2. Auto-shift if requested
        if auto_shift:
            self._auto_shift()

        # 3. Engine / drivetrain force
        engine_force = get_wheel_force(self.rpm, self.gear, throttle)

        # 4. Aerodynamic drag  (F_drag = ½ ρ Cd A v²)
        drag_force = AERO_DRAG_K * self.velocity ** 2

        # 5. Rolling resistance  (small, roughly constant)
        rolling_force = ROLLING_RESIST * CAR_MASS * 9.81

        # 6. Braking force
        brake_force = brake * MAX_BRAKE_FORCE

        # 7. Net force — resistances always oppose motion
        if self.velocity > 0.1:
            resistance = drag_force + rolling_force + brake_force
        elif self.velocity < -0.1:
            resistance = -(drag_force + rolling_force + brake_force)
        else:
            resistance = 0
            if abs(engine_force) < rolling_force + brake_force:
                engine_force = 0   # not enough force to move from rest

        net_force = engine_force - resistance

        # 8. F = ma
        acceleration = net_force / CAR_MASS

        # 9. Euler integration
        self.velocity = max(0.0, self.velocity + acceleration * dt)
        self.position += self.velocity * dt
        self.time     += dt

        # 10. Record
        self.history['time'].append(self.time)
        self.history['velocity_ms'].append(self.velocity)
        self.history['velocity_kmh'].append(self.velocity * 3.6)
        self.history['rpm'].append(self.rpm)
        self.history['gear'].append(self.gear)
        self.history['position'].append(self.position)
        self.history['engine_force'].append(engine_force)
        self.history['drag_force'].append(drag_force)
        self.history['net_force'].append(net_force)
        self.history['acceleration'].append(acceleration)
