import numpy as np

class Car:
    def __init__(self, position, heading=0.0, velocity=0.0):
        self.position = np.array(position, dtype=float)
        self.heading = heading
        self.velocity = velocity
        self.width = 1.9
        self.length = 3.9
        self.tire_width = 0.29
        self.tire_length = 0.705
        

    def get_corners(self):
        cx, cy = self.position
        L = self.length / 2
        W = self.width / 2


        corners = np.array([
            [ L,  W],
            [ L, -W],
            [-L, -W],
            [-L,  W]
        ])

        # rotation
        c = np.cos(self.heading)
        s = np.sin(self.heading)

        R = np.array([
            [c, -s],
            [s,  c]
        ])

        rotated = (R @ corners.T).T
        rotated[:, 0] += cx
        rotated[:, 1] += cy

        return rotated
    
    def get_tire_bottom_left_corners(self):
        cx, cy = self.position
        L = self.length / 2
        W = self.width / 2


        corners = np.array([
            [ L - self.tire_length,  W - self.tire_width],
            [ L - self.tire_length, -W ],
            [-L , -W ],
            [-L ,  W - self.tire_width]
        ])


        # rotation
        c = np.cos(self.heading)
        s = np.sin(self.heading)

        R = np.array([
            [c, -s],
            [s,  c]
        ])

        rotated = (R @ corners.T).T
        rotated[:, 0] += cx
        rotated[:, 1] += cy

        return rotated


class newCar():
    GEAR_RATIOS = {
        'R': -4.50, 
        'N': 0.00,
        1: 7.00, 
        2: 5.00, 
        3: 3.75, 
        4: 2.95,
        5: 2.40, 
        6: 1.98, 
        7: 1.71, 
        8: 1.52 }
    FINAL_DRIVE_RATIO = 3.27
    WHEEL_RADIUS       = 0.360   # m
    CAR_MASS           = 798     # kg
    RPM_IDLE           = 800
    RPM_REDLINE        = 15000
    AERO_DRAG_K        = 0.5 * 0.7 * 1.5 * 1.225   # 0.5 * Cd * A * rho
    ROLLING_RESIST     = 0.015                        # dimensionless coefficient
    MAX_BRAKE_FORCE    = 25000   # will need to update once downforace is implemented
    ENGINE_BRAKE_COEFF = 1500
    TORQUE_A = 50       # Nm baseline
    TORQUE_B = 380      # Nm peak above baseline
    TORQUE_C = 1.0e-8   # curve width (narrowness)
    TORQUE_D = 11000    # RPM at peak torque

    def __init__(self, position : tuple, heading : float, velocity : float):
        self.position = np.array(position, dtype=float) #current centered position, updated outside of this class, car is rectangle
        self.heading = heading #in radians, 0 means facing right
        self.velocity = velocity #current velocity, updated outside of this class
        self.width = 2.0 #fixed
        self.length = 5.63 #fixed
        self.tire_width = 0.305 #fixed
        self.tire_length = 0.72 #fixed
        self.tire_angle = 0.0 #current angle, from -15 to 15 degrees, updated outside of this class
        self.throttle = 0.0 #current throttle, from 0 do 1, updated outside of this class
        self.brake = 0.0 #current brake, from 0 to 1, updated outside of this class
        self.steering = 0.0 #current steering, from -180 to 180 degrees, updated outside of this class
        self.steering_force = 0.0 #current steering force, from -1 to 1, updated outside of this class
        self.wheel_base = 3.6 #fixed
        self.throttle_force = 0.0 #current throttle force, from 0 to 1, updated outside of this class
        self.brake_force = 0.0 #current brake force, from 0 to 1, updated outside of this class
        self.gear = 1
        self.rpm = self.RPM_IDLE

    def update_position(self, dt):
        self._update_steering()
        self._update_heading(dt)
        self._update_pedals()
        self._update_velocity(dt)

        self.position[0] += self.velocity * np.cos(self.heading) * dt
        self.position[1] += self.velocity * np.sin(self.heading) * dt
        return

    def _update_steering(self):
        self.steering += self.steering_force * 2 * (180.0 / 60.0)
        self.steering -= np.sign(self.steering) * (180.0 / 60.0)
        self.steering  = np.clip(self.steering, -180.0, 180.0)
        self.tire_angle = self.steering / 12.0
        return
    

    def _update_heading(self, dt):
        self.heading += (self.velocity / self.wheel_base) * np.tan(np.deg2rad(self.tire_angle)) * dt
        return 
    
    def _update_pedals(self):
        if self.brake > 0.0:
            self.throttle_force = 0.0
        self.throttle += self.throttle_force * (2.0 / 60.0) - (1.0 / 60.0)
        self.throttle  = np.clip(self.throttle, 0.0, 1.0)
        self.brake    += self.brake_force    * (2.0 / 60.0) - (1.0 / 60.0)
        self.brake     = np.clip(self.brake, 0.0, 1.0)
        return


    def _update_velocity(self, dt):
        self.rpm = self._rpm_from_velocity(self.velocity, self.gear)
        self._auto_shift()
        gear_ratio    = self.GEAR_RATIOS[self.gear]
        engine_torque = self._torque_curve(self.rpm)
        wheel_torque  = engine_torque * abs(gear_ratio) * self.FINAL_DRIVE_RATIO
        engine_force  = (wheel_torque / self.WHEEL_RADIUS) * self.throttle
        engine_brake  = self.ENGINE_BRAKE_COEFF * (self.rpm / self.RPM_REDLINE) if self.throttle < 0.05 else 0.0
        drag_force    = self.AERO_DRAG_K * self.velocity ** 2
        rolling_force = self.ROLLING_RESIST * self.CAR_MASS * 9.81
        brake_force   = self.brake * self.MAX_BRAKE_FORCE
        if self.velocity > 0.1:
            resistance = drag_force + rolling_force + brake_force + engine_brake
        else:
            resistance = 0.0
            if engine_force < rolling_force + brake_force:
                engine_force = 0.0
        net_force    = engine_force - resistance
        acceleration = net_force / self.CAR_MASS
        self.velocity = max(0.0, self.velocity + acceleration * dt)
        return
    
    def _torque_curve(self, rpm):
        return self.TORQUE_A + self.TORQUE_B * np.exp(-self.TORQUE_C * (rpm - self.TORQUE_D) ** 2)

    def _rpm_from_velocity(self, velocity, gear):
        if gear == 'N' or velocity <= 0:
            return self.RPM_IDLE
        wheel_omega  = velocity / self.WHEEL_RADIUS
        engine_omega = wheel_omega * self.GEAR_RATIOS[gear] * self.FINAL_DRIVE_RATIO
        rpm = engine_omega * (60 / (2 * np.pi))
        return float(np.clip(rpm, self.RPM_IDLE, self.RPM_REDLINE))

    def _auto_shift(self):
        if self.rpm >= 12000 and self.gear < 8:
            self.gear += 1
        elif self.rpm <= 8000 and self.gear > 1:
            self.gear -= 1
