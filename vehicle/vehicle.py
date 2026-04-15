import numpy as np


class newCar():

    # --- Drivetrain ---
    GEAR_RATIOS = {
        'R': -4.50, 'N': 0.00,
        1: 7.00, 2: 5.00, 3: 3.75, 4: 2.95,
        5: 2.40, 6: 1.98, 7: 1.71, 8: 1.52,
    }
    FINAL_DRIVE_RATIO = 3.27
    WHEEL_RADIUS      = 0.360   # m

    # --- Engine ---
    RPM_IDLE           = 800
    RPM_REDLINE        = 15000
    TORQUE_A           = 50      # Nm  — baseline torque
    TORQUE_B           = 380     # Nm  — peak above baseline
    TORQUE_C           = 1.0e-8  # 1/RPM²  — curve width
    TORQUE_D           = 11000   # RPM — at peak torque
    ENGINE_BRAKE_COEFF = 1500    # N   — max engine braking force

    # --- Mass ---
    CAR_MASS = 798  # kg

    # --- Aerodynamics (0.5 * C * A * rho) ---
    AERO_DRAG_K      = 0.5 * 0.7 * 1.5 * 1.225  # drag
    AERO_DOWNFORCE_K = 0.5 * 3.5 * 1.5 * 1.225  # downforce

    # --- Friction ---
    ROLLING_RESIST = 0.015  # rolling resistance coefficient
    BRAKE_FRICTION = 1.8    # brake friction coefficient

    # ------------------------------------------------------------------ #

    def __init__(self, position: tuple, heading: float, velocity: float):

        # Kinematic state
        self.position = np.array(position, dtype=float)  # centre, m
        self.heading  = heading   # radians, 0 = facing right
        self.velocity = velocity  # m/s, longitudinal

        # Body geometry
        self.width       = 2.0
        self.length      = 5.63
        self.wheel_base  = 3.6
        self.tire_width  = 0.305
        self.tire_length = 0.72

        # Control inputs — set each frame by the caller
        self.throttle_force = 0.0   # [0, 1]
        self.brake_force    = 0.0   # [0, 1]
        self.steering_force = 0.0   # [-1, 1]

        # Pedal state
        self.throttle = 0.0  # [0, 1]
        self.brake    = 0.0  # [0, 1]

        # Steering / tyre state
        self.steering   = 0.0  # accumulator, degrees [-180, 180]
        self.tire_angle = 0.0  # front-wheel angle, degrees [-15, 15]

        # Drivetrain state
        self.gear = 1
        self.rpm  = self.RPM_IDLE

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def update_position(self, dt):
        self._update_steering()
        self._update_heading(dt)
        self._update_pedals()
        self._update_velocity(dt)
        self.position[0] += self.velocity * np.cos(self.heading) * dt
        self.position[1] += self.velocity * np.sin(self.heading) * dt

    # ------------------------------------------------------------------ #
    # Physics steps                                                        #
    # ------------------------------------------------------------------ #

    def _update_steering(self):
        self.steering += self.steering_force * 2 * (180.0 / 60.0)
        self.steering -= np.sign(self.steering) * (180.0 / 60.0)
        self.steering  = np.clip(self.steering, -180.0, 180.0)
        self.tire_angle = self.steering / 12.0

    def _update_heading(self, dt):
        # Kinematic bicycle model
        self.heading += (self.velocity / self.wheel_base) * np.tan(np.deg2rad(self.tire_angle)) * dt

    def _update_pedals(self):
        if self.brake > 0.0:
            self.throttle_force = 0.0
        self.throttle += self.throttle_force * (2.0 / 60.0) - (1.0 / 60.0)
        self.throttle  = np.clip(self.throttle, 0.0, 1.0)
        self.brake    += self.brake_force    * (2.0 / 60.0) - (1.0 / 60.0)
        self.brake     = np.clip(self.brake, 0.0, 1.0)

    def _update_velocity(self, dt):
        self.rpm = self._rpm_from_velocity(self.velocity, self.gear)
        self._auto_shift()

        engine_force  = (self._torque_curve(self.rpm) * abs(self.GEAR_RATIOS[self.gear]) * self.FINAL_DRIVE_RATIO / self.WHEEL_RADIUS) * self.throttle
        engine_brake  = self.ENGINE_BRAKE_COEFF * (self.rpm / self.RPM_REDLINE) if self.throttle < 0.05 else 0.0

        downforce     = self.AERO_DOWNFORCE_K * self.velocity ** 2
        drag_force    = self.AERO_DRAG_K      * self.velocity ** 2
        rolling_force = self.ROLLING_RESIST   * self.CAR_MASS * 9.81
        brake_force   = self.brake * (self.CAR_MASS * 9.81 + downforce) * self.BRAKE_FRICTION

        if self.velocity > 0.1:
            resistance = drag_force + rolling_force + brake_force + engine_brake
        else:
            resistance = 0.0
            if engine_force < rolling_force + brake_force:
                engine_force = 0.0

        self.velocity = max(0.0, self.velocity + (engine_force - resistance) / self.CAR_MASS * dt)

    # ------------------------------------------------------------------ #
    # Engine helpers                                                       #
    # ------------------------------------------------------------------ #

    def _torque_curve(self, rpm):
        return self.TORQUE_A + self.TORQUE_B * np.exp(-self.TORQUE_C * (rpm - self.TORQUE_D) ** 2)

    def _rpm_from_velocity(self, velocity, gear):
        if gear == 'N' or velocity <= 0:
            return self.RPM_IDLE
        wheel_omega  = velocity / self.WHEEL_RADIUS
        engine_omega = wheel_omega * self.GEAR_RATIOS[gear] * self.FINAL_DRIVE_RATIO
        return float(np.clip(engine_omega * (60 / (2 * np.pi)), self.RPM_IDLE, self.RPM_REDLINE))

    def _auto_shift(self):
        if self.rpm >= 12000 and self.gear < 8:
            self.gear += 1
        elif self.rpm <= 8000 and self.gear > 1:
            self.gear -= 1
