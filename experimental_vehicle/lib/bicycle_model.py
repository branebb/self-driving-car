"""
Bicycle Model — 2-DOF Planar Vehicle Dynamics
==============================================

Provides two classes:

  BicycleModel
      Pure-lateral bicycle model from notebook 03.
      Constant Vx; tire forces from Pacejka lateral only.
      Good for studying understeer/oversteer balance.

  CombinedBicycleModel
      Adds braking input and full combined slip tire forces.
      Vx is now dynamic — the car can decelerate.
      Uses combined_slip_forces() from combined_slip.py so that braking
      while cornering correctly reduces lateral grip (friction circle).

Both classes share the same F1 vehicle parameters (2024 rules):
  m = 798 kg, Iz = 450 kg·m², a = 1.94 m, b = 1.66 m
"""

import numpy as np
from pacejka       import pacejka
from combined_slip import combined_slip_forces


# ── Shared F1 vehicle constants ───────────────────────────────────────────────

M    = 798.0              # kg   total mass (car + driver)
IZ   = 450.0              # kg·m²  yaw moment of inertia (very low for F1)
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

        speed   = np.sqrt(self.Vx**2 + self.Vy**2)
        self.x += speed * np.cos(self.psi) * dt
        self.y += speed * np.sin(self.psi) * dt
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
        speed   = np.sqrt(self.Vx**2 + self.Vy**2)
        self.x += speed * np.cos(self.psi) * dt
        self.y += speed * np.sin(self.psi) * dt
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
