"""
Bicycle Model — 2-DOF Planar Vehicle Dynamics
==============================================

Provides four classes:

  BicycleModel
      Pure-lateral bicycle model from notebook 03.
      Constant Vx; tire forces from Pacejka lateral only.
      Good for studying understeer/oversteer balance.

  CombinedBicycleModel
      Adds braking input and full combined slip tire forces.
      Vx is now dynamic — the car can decelerate.
      Uses combined_slip_forces() from combined_slip.py so that braking
      while cornering correctly reduces lateral grip (friction circle).

  WeightTransferModel
      Adds dynamic per-tire normal loads on top of CombinedBicycleModel.
      Under acceleration/braking, Fz shifts front–rear (longitudinal WT).
      Under cornering, Fz shifts left–right per axle (lateral WT).
      Because D(Fz) is sublinear, weight transfer always reduces total grip
      (this is the load-sensitivity penalty explored in notebook 05).

  AeroBicycleModel
      Adds aerodynamic downforce and drag on top of WeightTransferModel.
      Downforce adds to per-axle Fz before the lateral load split, boosting
      grip at high speed.  Drag subtracts from the longitudinal force budget.
      Three setup variants (Monaco / Silverstone / Monza) are parameterised
      by C_DF and C_DR (notebook 06).

All classes share the same F1 vehicle parameters (2024 rules):
  m = 798 kg, Iz = 800 kg·m², a = 1.94 m, b = 1.66 m
"""

import numpy as np
from pacejka       import pacejka
from combined_slip import combined_slip_forces
from engine        import (
    torque_curve, GEAR_RATIOS, FINAL_DRIVE_RATIO,
    WHEEL_RADIUS, RPM_IDLE, RPM_REDLINE,
    ROLLING_RESIST,
)


# ── Shared F1 vehicle constants ───────────────────────────────────────────────

M    = 798.0              # kg   total mass (car + driver)
IZ   = 800.0              # kg·m²  yaw moment of inertia (real F1 estimate: 700-1000)
A    = 1.94               # m   CG to front axle   (46 % front weight distribution)
B    = 1.66               # m   CG to rear axle    (54 % rear weight distribution)
FZ_F = M * 9.81 * 0.46   # N   static normal load, front axle
FZ_R = M * 9.81 * 0.54   # N   static normal load, rear axle

# Pacejka shape parameters (same for both classes)
B_LAT = 10.0
C_LAT =  1.90
B_LON = 12.0
C_LON =  1.65
E_PAC =  0.97

# Braking parameters for CombinedBicycleModel
KAPPA_AT_FULL_BRAKE = 0.15   # peak slip ratio at full braking (ABS-managed)
BRAKE_BIAS_F        = 0.60   # fraction of brake force on front axle (F1 typical)


# ── BicycleModel ──────────────────────────────────────────────────────────────

class BicycleModel:
    """
    Non-linear bicycle model with Pacejka lateral tires.

    Constant Vx — no drive or braking forces, just cornering.
    Reproduces notebook 03 inline implementation for use in later notebooks.

    State: Vy, psi_dot, psi, x, y
    Input: delta (steer angle, rad)
    """

    M   = M;    IZ  = IZ
    A   = A;    B   = B
    FZ_F = FZ_F; FZ_R = FZ_R
    B_PAC = B_LAT; C_PAC = C_LAT; E_PAC = E_PAC

    def __init__(self, Vx=30.0, D_front=1.6, D_rear=1.6):
        self.Vx      = Vx
        self.D_front = D_front
        self.D_rear  = D_rear

        self.Vy      = 0.0
        self.psi     = 0.0
        self.psi_dot = 0.0
        self.x       = 0.0
        self.y       = 0.0
        self.t       = 0.0

        self.history = {
            't': [], 'x': [], 'y': [], 'psi': [], 'psi_dot': [],
            'Vy': [], 'alpha_f': [], 'alpha_r': [], 'Fy_f': [], 'Fy_r': [],
            'delta': [],
        }

    def _slip_angles(self, delta):
        Vx, Vy, pd = self.Vx, self.Vy, self.psi_dot
        if Vx < 0.5:
            return 0.0, 0.0
        alpha_f = delta - np.arctan2(Vy + self.A  * pd, Vx)
        alpha_r =       - np.arctan2(Vy - self.B  * pd, Vx)
        return alpha_f, alpha_r

    def _lateral_force(self, alpha, D, Fz):
        mu = pacejka(alpha, self.B_PAC, self.C_PAC, D, self.E_PAC)
        return mu * Fz

    def step(self, delta, dt=0.005):
        alpha_f, alpha_r = self._slip_angles(delta)

        Fy_f = self._lateral_force(alpha_f, self.D_front, self.FZ_F)
        Fy_r = self._lateral_force(alpha_r, self.D_rear,  self.FZ_R)

        dVy       = (Fy_f + Fy_r) / self.M - self.Vx * self.psi_dot
        d_psi_dot = (self.A * Fy_f - self.B * Fy_r) / self.IZ

        self.Vy      += dVy       * dt
        self.psi_dot += d_psi_dot * dt
        self.psi     += self.psi_dot * dt

        self.x += (self.Vx * np.cos(self.psi) - self.Vy * np.sin(self.psi)) * dt
        self.y += (self.Vx * np.sin(self.psi) + self.Vy * np.cos(self.psi)) * dt
        self.t += dt

        self.history['t'].append(self.t)
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['psi'].append(self.psi)
        self.history['psi_dot'].append(self.psi_dot)
        self.history['Vy'].append(self.Vy)
        self.history['alpha_f'].append(np.degrees(alpha_f))
        self.history['alpha_r'].append(np.degrees(alpha_r))
        self.history['Fy_f'].append(Fy_f)
        self.history['Fy_r'].append(Fy_r)
        self.history['delta'].append(np.degrees(delta))


# ── CombinedBicycleModel ──────────────────────────────────────────────────────

class CombinedBicycleModel:
    """
    Bicycle model with combined slip tires and dynamic Vx.

    Extends BicycleModel with:
      - braking input → longitudinal slip κ at each axle
      - combined_slip_forces() for all tire forces (Fx AND Fy coupled)
      - Vx is now a state variable (braking decelerates the car)

    Equations of motion (body frame):
      m  · dVx/dt         = Fx_f + Fx_r          (longitudinal, Coriolis term omitted)
      m  · (dVy/dt + Vx·ψ̇) = Fy_f + Fy_r          (lateral)
      Iz · dψ̇/dt           = a·Fy_f − b·Fy_r       (yaw)

    Braking model:
      brake ∈ [0, 1] maps linearly to slip ratio:
        κ_f = −brake × KAPPA_AT_FULL_BRAKE × BRAKE_BIAS_F
        κ_r = −brake × KAPPA_AT_FULL_BRAKE × (1 − BRAKE_BIAS_F)
      This approximates ABS-managed braking near the peak slip ratio.
    """

    M    = M;    IZ   = IZ
    A    = A;    B    = B
    FZ_F = FZ_F; FZ_R = FZ_R
    B_LON = B_LON; C_LON = C_LON
    B_LAT = B_LAT; C_LAT = C_LAT
    E_PAC = E_PAC
    KAPPA_AT_FULL_BRAKE = KAPPA_AT_FULL_BRAKE
    BRAKE_BIAS_F        = BRAKE_BIAS_F

    def __init__(self, Vx=30.0, D_front=1.6, D_rear=1.6):
        self.Vx      = float(Vx)
        self.D_front = D_front
        self.D_rear  = D_rear

        self.Vy      = 0.0
        self.psi     = 0.0
        self.psi_dot = 0.0
        self.x       = 0.0
        self.y       = 0.0
        self.t       = 0.0

        self.history = {
            't': [], 'x': [], 'y': [], 'psi': [], 'psi_dot': [],
            'Vx': [], 'Vy': [],
            'alpha_f': [], 'alpha_r': [],
            'kappa_f': [], 'kappa_r': [],
            'Fx_f': [], 'Fx_r': [], 'Fy_f': [], 'Fy_r': [],
            'brake': [], 'delta': [],
        }

    def _slip_angles(self, delta):
        Vx, Vy, pd = self.Vx, self.Vy, self.psi_dot
        if Vx < 0.5:
            return 0.0, 0.0
        alpha_f = delta - np.arctan2(Vy + self.A * pd, Vx)
        alpha_r =       - np.arctan2(Vy - self.B * pd, Vx)
        return alpha_f, alpha_r

    def step(self, delta, brake=0.0, dt=0.005):
        """
        Advance simulation by dt seconds.

        Parameters
        ----------
        delta : float   Steer angle (rad), positive = left
        brake : float   Brake pedal 0 – 1, linear map to slip ratio
        dt    : float   Timestep (s), default 5 ms
        """
        # 1. Lateral slip angles from steering + yaw dynamics
        alpha_f, alpha_r = self._slip_angles(delta)

        # 2. Longitudinal slip from braking (negative = braking)
        kappa_f = -brake * self.KAPPA_AT_FULL_BRAKE * self.BRAKE_BIAS_F
        kappa_r = -brake * self.KAPPA_AT_FULL_BRAKE * (1.0 - self.BRAKE_BIAS_F)

        # 3. Combined slip forces — friction circle is respected automatically
        Fx_f, Fy_f = combined_slip_forces(
            kappa_f, alpha_f, self.FZ_F,
            self.B_LON, self.C_LON, self.D_front, self.E_PAC,
            self.B_LAT, self.C_LAT, self.D_front, self.E_PAC,
        )
        Fx_r, Fy_r = combined_slip_forces(
            kappa_r, alpha_r, self.FZ_R,
            self.B_LON, self.C_LON, self.D_rear,  self.E_PAC,
            self.B_LAT, self.C_LAT, self.D_rear,  self.E_PAC,
        )

        # 4. Equations of motion (body frame)
        dVx       = (Fx_f + Fx_r) / self.M                          # longitudinal
        dVy       = (Fy_f + Fy_r) / self.M - self.Vx * self.psi_dot # lateral + centripetal
        d_psi_dot = (self.A * Fy_f - self.B * Fy_r) / self.IZ       # yaw

        # 5. Euler integration
        self.Vx       = max(0.5, self.Vx + dVx * dt)  # floor at 0.5 m/s
        self.Vy      += dVy       * dt
        self.psi_dot += d_psi_dot * dt
        self.psi     += self.psi_dot * dt

        # 6. World-frame position
        self.x += (self.Vx * np.cos(self.psi) - self.Vy * np.sin(self.psi)) * dt
        self.y += (self.Vx * np.sin(self.psi) + self.Vy * np.cos(self.psi)) * dt
        self.t += dt

        # 7. Record
        self.history['t'].append(self.t)
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['psi'].append(self.psi)
        self.history['psi_dot'].append(self.psi_dot)
        self.history['Vx'].append(self.Vx)
        self.history['Vy'].append(self.Vy)
        self.history['alpha_f'].append(np.degrees(alpha_f))
        self.history['alpha_r'].append(np.degrees(alpha_r))
        self.history['kappa_f'].append(kappa_f)
        self.history['kappa_r'].append(kappa_r)
        self.history['Fx_f'].append(Fx_f)
        self.history['Fx_r'].append(Fx_r)
        self.history['Fy_f'].append(Fy_f)
        self.history['Fy_r'].append(Fy_r)
        self.history['brake'].append(brake)
        self.history['delta'].append(np.degrees(delta))


# ── Additional constants for WeightTransferModel ──────────────────────────────

G   = 9.81      # m/s²  gravitational acceleration
HCG = 0.30      # m     CG height above ground (F1 cars are extremely low)
TF  = 1.60      # m     front track width (contact-patch centre to centre)
TR  = 1.60      # m     rear  track width
WB  = A + B     # m     wheelbase  (1.94 + 1.66 = 3.60 m)

# Load-sensitivity: peak friction coefficient  μ(Fz) = D1 − D2·Fz
# Calibrated: μ = 1.60 at Fz = 1 800 N,  μ = 1.35 at Fz = 3 600 N
D1 = 1.85       # dimensionless   linear  term
D2 = 1.39e-4    # N⁻¹             quadratic term  (sublinearity / saturation)


# ── WeightTransferModel ────────────────────────────────────────────────────────

class WeightTransferModel:
    """
    Bicycle model with combined slip tires and dynamic per-tire normal loads.

    Extends CombinedBicycleModel by tracking individual wheel loads and
    updating them every timestep from longitudinal + lateral weight transfer.

    Weight transfer formulas  (rigid body, no suspension compliance):
      dFz_lon   = M · ax · hcg / WB         (+ ax = accel → rear loads up)
      dFz_lat_f = Fz_f · ay · hcg / (g·Tf) (+ ay = left-turn → right side loads up)
      dFz_lat_r = Fz_r · ay · hcg / (g·Tr)

    Per-tire normal loads:
      Fz_fl = (FZ_F − dFz_lon)/2 − dFz_lat_f
      Fz_fr = (FZ_F − dFz_lon)/2 + dFz_lat_f
      Fz_rl = (FZ_R + dFz_lon)/2 − dFz_lat_r
      Fz_rr = (FZ_R + dFz_lon)/2 + dFz_lat_r

    Each wheel uses its own load-sensitive peak friction:
      μ(Fz) = D1 − D2·Fz

    Because μ(Fz) is sublinear, weight transfer always reduces total axle grip
    (the load-sensitivity penalty).  Rear axle starts with higher static Fz per
    wheel (2 100 N vs 1 800 N), so its μ is already slightly lower — and under
    lateral WT it loses proportionally more grip than the front.

    One-step lag: loads are updated using accelerations from the PREVIOUS step
    to avoid an algebraic loop (forces depend on loads depend on forces).
    """

    def __init__(self, Vx=30.0):
        self.Vx      = float(Vx)
        self.Vy      = 0.0
        self.psi     = 0.0
        self.psi_dot = 0.0
        self.x       = 0.0
        self.y       = 0.0
        self.t       = 0.0

        # Per-tire normal loads — initialise to static values
        self.Fz_fl = FZ_F / 2.0
        self.Fz_fr = FZ_F / 2.0
        self.Fz_rl = FZ_R / 2.0
        self.Fz_rr = FZ_R / 2.0

        # Previous-step inertial accelerations (for one-step-lag load update)
        self._ax = 0.0
        self._ay = 0.0

        self.history = {
            't':       [], 'x':       [], 'y':       [],
            'psi':     [], 'psi_dot': [],
            'Vx':      [], 'Vy':      [],
            'alpha_f': [], 'alpha_r': [],
            'kappa_f': [], 'kappa_r': [],
            'Fx_f':    [], 'Fx_r':    [],
            'Fy_f':    [], 'Fy_r':    [],
            'brake':   [], 'delta':   [],
            'Fz_fl':   [], 'Fz_fr':   [],
            'Fz_rl':   [], 'Fz_rr':   [],
            'ax':      [], 'ay':      [],
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _slip_angles(self, delta):
        if self.Vx < 0.5:
            return 0.0, 0.0
        alpha_f = delta - np.arctan2(self.Vy + A * self.psi_dot, self.Vx)
        alpha_r =       - np.arctan2(self.Vy - B * self.psi_dot, self.Vx)
        return alpha_f, alpha_r

    def _update_loads(self):
        """Recompute per-tire loads from previous-step inertial accelerations."""
        ax, ay = self._ax, self._ay

        # Longitudinal transfer: positive ax (accel) → rear gains, front loses
        dFz_lon = M * ax * HCG / WB

        # Total axle loads after longitudinal transfer
        Fz_f = FZ_F - dFz_lon
        Fz_r = FZ_R + dFz_lon

        # Lateral transfer: positive ay (left-turn force) → right side gains
        dFz_lat_f = Fz_f * ay * HCG / (G * TF)
        dFz_lat_r = Fz_r * ay * HCG / (G * TR)

        # Per-tire loads; clamped to 50 N (tyres cannot pull the car down)
        self.Fz_fl = max(50.0, Fz_f / 2.0 - dFz_lat_f)
        self.Fz_fr = max(50.0, Fz_f / 2.0 + dFz_lat_f)
        self.Fz_rl = max(50.0, Fz_r / 2.0 - dFz_lat_r)
        self.Fz_rr = max(50.0, Fz_r / 2.0 + dFz_lat_r)

    def _axle_forces(self, kappa, alpha, Fz_left, Fz_right):
        """
        Compute combined Fx, Fy for one axle by summing left + right wheels.
        Each wheel uses its own load-sensitive peak friction coefficient.
        """
        mu_l = max(0.5, D1 - D2 * Fz_left)
        mu_r = max(0.5, D1 - D2 * Fz_right)

        Fx_l, Fy_l = combined_slip_forces(
            kappa, alpha, Fz_left,
            B_LON, C_LON, mu_l, E_PAC,
            B_LAT, C_LAT, mu_l, E_PAC,
        )
        Fx_r, Fy_r = combined_slip_forces(
            kappa, alpha, Fz_right,
            B_LON, C_LON, mu_r, E_PAC,
            B_LAT, C_LAT, mu_r, E_PAC,
        )
        return Fx_l + Fx_r, Fy_l + Fy_r

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, delta, brake=0.0, dt=0.005):
        """
        Advance simulation by dt seconds.

        Parameters
        ----------
        delta : float   Steer angle (rad), positive = left
        brake : float   Brake pedal [0, 1]
        dt    : float   Timestep (s), default 5 ms
        """
        # 1. Update per-tire loads from previous-step accelerations
        self._update_loads()

        # 2. Slip angles
        alpha_f, alpha_r = self._slip_angles(delta)

        # 3. Longitudinal slip (ABS-managed, linear map to slip ratio)
        kappa_f = -brake * KAPPA_AT_FULL_BRAKE * BRAKE_BIAS_F
        kappa_r = -brake * KAPPA_AT_FULL_BRAKE * (1.0 - BRAKE_BIAS_F)

        # 4. Tire forces — per wheel with load-sensitive μ, summed per axle
        Fx_f, Fy_f = self._axle_forces(kappa_f, alpha_f, self.Fz_fl, self.Fz_fr)
        Fx_r, Fy_r = self._axle_forces(kappa_r, alpha_r, self.Fz_rl, self.Fz_rr)

        # 5. Inertial accelerations in body frame
        ax = (Fx_f + Fx_r) / M
        ay = (Fy_f + Fy_r) / M

        # 6. Store for next load update (one-step lag)
        self._ax = ax
        self._ay = ay

        # 7. Euler integration
        d_psi_dot = (A * Fy_f - B * Fy_r) / IZ
        self.Vx       = max(0.5, self.Vx + ax * dt)
        self.Vy      += (ay - self.Vx * self.psi_dot) * dt
        self.psi_dot += d_psi_dot * dt
        self.psi     += self.psi_dot * dt

        # 8. World-frame position
        self.x += (self.Vx * np.cos(self.psi) - self.Vy * np.sin(self.psi)) * dt
        self.y += (self.Vx * np.sin(self.psi) + self.Vy * np.cos(self.psi)) * dt
        self.t += dt

        # 9. Record history
        h = self.history
        h['t'].append(self.t);           h['x'].append(self.x)
        h['y'].append(self.y);           h['psi'].append(self.psi)
        h['psi_dot'].append(self.psi_dot)
        h['Vx'].append(self.Vx);         h['Vy'].append(self.Vy)
        h['alpha_f'].append(np.degrees(alpha_f))
        h['alpha_r'].append(np.degrees(alpha_r))
        h['kappa_f'].append(kappa_f);    h['kappa_r'].append(kappa_r)
        h['Fx_f'].append(Fx_f);          h['Fx_r'].append(Fx_r)
        h['Fy_f'].append(Fy_f);          h['Fy_r'].append(Fy_r)
        h['brake'].append(brake);        h['delta'].append(np.degrees(delta))
        h['Fz_fl'].append(self.Fz_fl);   h['Fz_fr'].append(self.Fz_fr)
        h['Fz_rl'].append(self.Fz_rl);   h['Fz_rr'].append(self.Fz_rr)
        h['ax'].append(ax);              h['ay'].append(ay)


# ── Additional constants for AeroBicycleModel ─────────────────────────────────

# Default setup: medium downforce (Silverstone-style)
C_DF          = 1.50   # N·s²/m²  downforce coefficient  (F_down = C_DF · Vx²)
C_DR          = 0.64   # N·s²/m²  drag coefficient        (F_drag = C_DR · Vx²)
AERO_BALANCE_F = 0.37  # fraction of downforce carried by front axle (37 % F / 63 % R)


# ── AeroBicycleModel ───────────────────────────────────────────────────────────

class AeroBicycleModel(WeightTransferModel):
    """
    Bicycle model with combined slip, weight transfer, and aerodynamics.

    Extends WeightTransferModel with two lumped aero forces:

      F_down = c_df · Vx²   (downforce, N — adds to tyre normal loads)
      F_drag = c_dr · Vx²   (drag,      N — subtracts from longitudinal budget)

    Downforce distribution:
      Front axle receives  aero_balance_f  × F_down
      Rear  axle receives  (1 − aero_balance_f) × F_down

    These are stacked on top of static + longitudinal-WT axle loads before the
    lateral WT split, so every tyre benefits from the extra grip.

    Drag enters the longitudinal equation of motion:
      ax = (Fx_f + Fx_r − F_drag) / M

    Three typical F1 setups:
      High DF  (Monaco)      : C_DF = 2.5, C_DR = 0.80
      Medium   (Silverstone) : C_DF = 1.5, C_DR = 0.64  ← default
      Low DF   (Monza)       : C_DF = 0.7, C_DR = 0.25

    At 60 m/s (216 km/h) with medium DF:
      F_down ≈ 5 400 N  (≈ 69 % of car weight added to tyres)
      F_drag ≈ 1 800 N
    This raises the lateral grip ceiling from ~1.6 g to ~2.3 g.
    """

    def __init__(self, Vx=30.0, c_df=C_DF, c_dr=C_DR, aero_balance_f=AERO_BALANCE_F):
        super().__init__(Vx)
        self.c_df          = c_df
        self.c_dr          = c_dr
        self.aero_balance_f = aero_balance_f

        # Extra history channels for aero forces
        self.history['F_down'] = []
        self.history['F_drag'] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_loads_aero(self, F_down):
        """
        Recompute per-tire loads including aerodynamic downforce.

        Identical to WeightTransferModel._update_loads() except that the
        downforce is split front/rear and added to axle loads before the
        lateral weight-transfer split.
        """
        ax, ay = self._ax, self._ay

        # Longitudinal weight transfer
        dFz_lon = M * ax * HCG / WB

        # Axle loads: static distribution + longitudinal WT + downforce
        F_down_f = self.aero_balance_f         * F_down
        F_down_r = (1.0 - self.aero_balance_f) * F_down

        Fz_f = FZ_F - dFz_lon + F_down_f
        Fz_r = FZ_R + dFz_lon + F_down_r

        # Lateral weight transfer (uses aero-enhanced axle loads)
        dFz_lat_f = Fz_f * ay * HCG / (G * TF)
        dFz_lat_r = Fz_r * ay * HCG / (G * TR)

        # Per-tire loads; clamped to 50 N
        self.Fz_fl = max(50.0, Fz_f / 2.0 - dFz_lat_f)
        self.Fz_fr = max(50.0, Fz_f / 2.0 + dFz_lat_f)
        self.Fz_rl = max(50.0, Fz_r / 2.0 - dFz_lat_r)
        self.Fz_rr = max(50.0, Fz_r / 2.0 + dFz_lat_r)

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, delta, brake=0.0, dt=0.005):
        """
        Advance simulation by dt seconds.

        Parameters
        ----------
        delta : float   Steer angle (rad), positive = left
        brake : float   Brake pedal [0, 1]
        dt    : float   Timestep (s), default 5 ms
        """
        # 1. Current aero forces (speed from previous step)
        F_down = self.c_df * self.Vx ** 2
        F_drag = self.c_dr * self.Vx ** 2

        # 2. Update per-tire loads (aero-aware, one-step lag for accelerations)
        self._update_loads_aero(F_down)

        # 3. Slip angles
        alpha_f, alpha_r = self._slip_angles(delta)

        # 4. Longitudinal slip (ABS-managed)
        kappa_f = -brake * KAPPA_AT_FULL_BRAKE * BRAKE_BIAS_F
        kappa_r = -brake * KAPPA_AT_FULL_BRAKE * (1.0 - BRAKE_BIAS_F)

        # 5. Tire forces — per wheel with load-sensitive μ, summed per axle
        Fx_f, Fy_f = self._axle_forces(kappa_f, alpha_f, self.Fz_fl, self.Fz_fr)
        Fx_r, Fy_r = self._axle_forces(kappa_r, alpha_r, self.Fz_rl, self.Fz_rr)

        # 6. Inertial accelerations — drag reduces longitudinal budget
        ax = (Fx_f + Fx_r - F_drag) / M
        ay = (Fy_f + Fy_r) / M

        # 7. Store for next load update (one-step lag)
        self._ax = ax
        self._ay = ay

        # 8. Euler integration
        d_psi_dot = (A * Fy_f - B * Fy_r) / IZ
        self.Vx       = max(0.5, self.Vx + ax * dt)
        self.Vy      += (ay - self.Vx * self.psi_dot) * dt
        self.psi_dot += d_psi_dot * dt
        self.psi     += self.psi_dot * dt

        # 9. World-frame position
        self.x += (self.Vx * np.cos(self.psi) - self.Vy * np.sin(self.psi)) * dt
        self.y += (self.Vx * np.sin(self.psi) + self.Vy * np.cos(self.psi)) * dt
        self.t += dt

        # 10. Record history
        h = self.history
        h['t'].append(self.t);           h['x'].append(self.x)
        h['y'].append(self.y);           h['psi'].append(self.psi)
        h['psi_dot'].append(self.psi_dot)
        h['Vx'].append(self.Vx);         h['Vy'].append(self.Vy)
        h['alpha_f'].append(np.degrees(alpha_f))
        h['alpha_r'].append(np.degrees(alpha_r))
        h['kappa_f'].append(kappa_f);    h['kappa_r'].append(kappa_r)
        h['Fx_f'].append(Fx_f);          h['Fx_r'].append(Fx_r)
        h['Fy_f'].append(Fy_f);          h['Fy_r'].append(Fy_r)
        h['brake'].append(brake);        h['delta'].append(np.degrees(delta))
        h['Fz_fl'].append(self.Fz_fl);   h['Fz_fr'].append(self.Fz_fr)
        h['Fz_rl'].append(self.Fz_rl);   h['Fz_rr'].append(self.Fz_rr)
        h['ax'].append(ax);              h['ay'].append(ay)
        h['F_down'].append(F_down);      h['F_drag'].append(F_drag)


# ── Additional constants for Full2DModel ──────────────────────────────────────

# Effective rear-wheel rotational inertia (wheel assembly + reflected drivetrain)
# Wheel + tyre: ~11 kg at r=0.33 m → I ≈ 0.60 kg·m²
# Brake disc:   ~ 3 kg at r=0.15 m → I ≈ 0.03 kg·m²
# Total per wheel ≈ 0.63; we use 1.5 to approximate reflected drivetrain inertia
I_WHEEL = 1.5   # kg·m²  per driven wheel (effective)


# ── Full2DModel ────────────────────────────────────────────────────────────────

class Full2DModel(AeroBicycleModel):
    """
    Complete 2D vehicle dynamics model.

    Extends AeroBicycleModel with a full drivetrain:

      throttle → engine torque → rear wheel angular velocity (ω_r)
        → traction slip ratio κ_r = (ω_r·R − Vx) / max(ω_r·R, Vx)
        → longitudinal tyre force Fx_r  (limited by Pacejka + load)
        → back-reaction on ω_r  (tyre force brakes the spinning wheel)

    This is a coupled feedback loop — the engine tries to spin the wheel up,
    the tyre force resists it.  At low speeds / high throttle the wheel
    overspeeds (wheelspin); at steady speed the system reaches equilibrium.

    Engine model:
      - Torque: Gaussian bell curve centred at 11 000 RPM
      - RPM:    back-computed from ω_r, gear ratio and final drive
      - Gear:   automatic (shift up at 90 % redline, down at 40 %)

    Braking:
      - ABS approximation (same as parent classes): κ = −brake × κ_peak × bias
      - During braking, ω_r is clamped to Vx/R (ABS prevents wheel lockup)

    Rolling resistance:
      - Constant force = ROLLING_RESIST × M × g opposing forward motion
      - Included in the longitudinal equation of motion

    Inputs per step: delta, throttle, brake  (all scalars, all continuous)
    New state variables: gear (int 1–8), rpm (float), omega_r (float, rad/s)

    Inheritance chain:
      BicycleModel
        └── CombinedBicycleModel
              └── WeightTransferModel
                    └── AeroBicycleModel
                          └── Full2DModel   ← this class
    """

    def __init__(self, Vx=10.0, gear=None,
                 c_df=C_DF, c_dr=C_DR, aero_balance_f=AERO_BALANCE_F):
        super().__init__(Vx, c_df, c_dr, aero_balance_f)

        # Pick starting gear so RPM is in a reasonable range
        if gear is None:
            gear = self._gear_for_speed(float(Vx))
        self.gear = gear

        # Rear wheel angular velocity — start at rolling (zero traction slip)
        self.omega_r = float(Vx) / WHEEL_RADIUS

        # Sync RPM to initial conditions
        self.rpm = self._compute_rpm()

        # Extra history channels
        self.history.update({
            'throttle':      [],
            'gear':          [],
            'rpm':           [],
            'kappa_r_drive': [],   # traction slip ratio (throttle side)
            'omega_r':       [],
        })

    # ── helpers ───────────────────────────────────────────────────────────────

    def _gear_for_speed(self, Vx_ms):
        """Return the lowest gear whose RPM stays below 85 % of redline."""
        for g in range(1, 9):
            rpm = (Vx_ms / WHEEL_RADIUS) * abs(GEAR_RATIOS[g]) * FINAL_DRIVE_RATIO \
                  * (60.0 / (2.0 * np.pi))
            if rpm < RPM_REDLINE * 0.85:
                return g
        return 8

    def _compute_rpm(self):
        """Back-compute engine RPM from current rear wheel speed and gear."""
        omega_engine = self.omega_r * abs(GEAR_RATIOS[self.gear]) * FINAL_DRIVE_RATIO
        rpm = omega_engine * (60.0 / (2.0 * np.pi))
        return float(np.clip(rpm, RPM_IDLE, RPM_REDLINE))

    def _auto_shift(self):
        """Shift up at 90 % redline; shift down below 40 % redline."""
        if self.rpm > RPM_REDLINE * 0.90 and self.gear < 8:
            self.gear += 1
        elif self.rpm < RPM_REDLINE * 0.40 and self.gear > 1:
            self.gear -= 1

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, delta, throttle=0.0, brake=0.0, dt=0.005, auto_shift=True):
        """
        Advance simulation by dt seconds.

        Parameters
        ----------
        delta    : float   Steer angle (rad), positive = left
        throttle : float   Throttle pedal [0, 1]
        brake    : float   Brake pedal   [0, 1]
        dt       : float   Timestep (s), default 5 ms
        auto_shift : bool  Automatic gear changes
        """
        # 1. Aero forces (speed from previous step)
        F_down = self.c_df * self.Vx ** 2
        F_drag = self.c_dr * self.Vx ** 2

        # 2. Rolling resistance (always opposes motion)
        F_roll = ROLLING_RESIST * M * G

        # 3. Update per-tire loads (aero-aware, one-step lag)
        self._update_loads_aero(F_down)

        # 4. Slip angles from steering + yaw dynamics
        alpha_f, alpha_r = self._slip_angles(delta)

        # 5. Traction slip ratio at rear wheels
        wheel_speed    = self.omega_r * WHEEL_RADIUS
        vx_ref         = max(abs(self.Vx), 0.5)
        kappa_r_drive  = (wheel_speed - vx_ref) / max(wheel_speed, vx_ref)
        # Clamp to physically reasonable range
        kappa_r_drive  = float(np.clip(kappa_r_drive, -0.5, 0.5))

        # 6. Effective slip ratios: throttle path vs braking path
        if brake > 0 and throttle == 0.0:
            # ABS braking: simplified slip model from parent classes
            kappa_f = -brake * KAPPA_AT_FULL_BRAKE * BRAKE_BIAS_F
            kappa_r = -brake * KAPPA_AT_FULL_BRAKE * (1.0 - BRAKE_BIAS_F)
            # Keep omega_r consistent with ABS (no lockup)
            self.omega_r = vx_ref / WHEEL_RADIUS
        else:
            kappa_f = 0.0
            kappa_r = kappa_r_drive

        # 7. Tyre forces — per wheel with load-sensitive μ, summed per axle
        Fx_f, Fy_f = self._axle_forces(kappa_f, alpha_f, self.Fz_fl, self.Fz_fr)
        Fx_r, Fy_r = self._axle_forces(kappa_r, alpha_r, self.Fz_rl, self.Fz_rr)

        # 8. Engine torque at rear wheels
        T_engine      = torque_curve(self.rpm) * abs(GEAR_RATIOS[self.gear]) * FINAL_DRIVE_RATIO
        T_drive       = T_engine * throttle   # total torque at rear axle

        # 9. Rear wheel rotational dynamics
        #    I · dω/dt = T_drive_per_wheel − F_tyre_lon_per_wheel · R
        #    (Fx_r is the total rear axle force → split by 2 per wheel)
        T_drive_per_wheel  = T_drive / 2.0
        Fx_r_per_wheel     = Fx_r    / 2.0
        d_omega_r = (T_drive_per_wheel - Fx_r_per_wheel * WHEEL_RADIUS) / I_WHEEL
        if brake == 0.0:   # only integrate ω when not under ABS control
            self.omega_r = max(0.0, self.omega_r + d_omega_r * dt)

        # 10. RPM + auto-shift
        self.rpm = self._compute_rpm()
        if auto_shift:
            self._auto_shift()
            self.rpm = self._compute_rpm()   # re-sync after possible gear change

        # 11. Inertial accelerations — drag and rolling resistance oppose motion
        ax = (Fx_f + Fx_r - F_drag - F_roll) / M
        ay = (Fy_f + Fy_r) / M

        # 12. Store for next load update (one-step lag)
        self._ax = ax
        self._ay = ay

        # 13. Euler integration
        d_psi_dot = (A * Fy_f - B * Fy_r) / IZ
        self.Vx       = max(0.5, self.Vx + ax * dt)
        self.Vy      += (ay - self.Vx * self.psi_dot) * dt
        self.psi_dot += d_psi_dot * dt
        self.psi     += self.psi_dot * dt

        # 14. World-frame position
        self.x += (self.Vx * np.cos(self.psi) - self.Vy * np.sin(self.psi)) * dt
        self.y += (self.Vx * np.sin(self.psi) + self.Vy * np.cos(self.psi)) * dt
        self.t += dt

        # 15. Record history
        h = self.history
        h['t'].append(self.t);              h['x'].append(self.x)
        h['y'].append(self.y);              h['psi'].append(self.psi)
        h['psi_dot'].append(self.psi_dot)
        h['Vx'].append(self.Vx);            h['Vy'].append(self.Vy)
        h['alpha_f'].append(np.degrees(alpha_f))
        h['alpha_r'].append(np.degrees(alpha_r))
        h['kappa_f'].append(kappa_f);       h['kappa_r'].append(kappa_r)
        h['Fx_f'].append(Fx_f);             h['Fx_r'].append(Fx_r)
        h['Fy_f'].append(Fy_f);             h['Fy_r'].append(Fy_r)
        h['brake'].append(brake);           h['delta'].append(np.degrees(delta))
        h['Fz_fl'].append(self.Fz_fl);      h['Fz_fr'].append(self.Fz_fr)
        h['Fz_rl'].append(self.Fz_rl);      h['Fz_rr'].append(self.Fz_rr)
        h['ax'].append(ax);                 h['ay'].append(ay)
        h['F_down'].append(F_down);         h['F_drag'].append(F_drag)
        h['throttle'].append(throttle)
        h['gear'].append(self.gear)
        h['rpm'].append(self.rpm)
        h['kappa_r_drive'].append(kappa_r_drive)
        h['omega_r'].append(self.omega_r)
