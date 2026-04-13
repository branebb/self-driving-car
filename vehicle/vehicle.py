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
    def __init__(self, position : tuple, heading : float, velocity : float):
        self.position = np.array(position, dtype=float) #current centered position, updated outside of this class, car is rectangle
        self.heading = heading #in radians, 0 means facing right
        self.velocity = velocity #current velocity, updated outside of this class
        self.width = 1.9 #fixed
        self.length = 3.9 #fixed
        self.tire_width = 0.29 #fixed
        self.tire_length = 0.705 #fixed
        self.tire_angle = 0.0 #current angle, from -15 to 15 degrees, updated outside of this class
        self.throttle = 0.0 #current throttle, from 0 do 1, updated outside of this class
        self.brake = 0.0 #current brake, from 0 to 1, updated outside of this class
        self.steering = 0.0 #current steering, from -180 to 180 degrees, updated outside of this class
        self.steering_force = 0.0 #current steering force, from -1 to 1, updated outside of this class
        self.wheel_base = 1.9 #fixed
        self.throttle_force = 0.0 #current throttle force, from 0 to 1, updated outside of this class
        self.brake_force = 0.0 #current brake force, from 0 to 1, updated outside of this class

    def update_position(self, dt):
        #steering update
        self.steering += self.steering_force * 2 * (180.0 / 60.0)
        self.steering -= np.sign(self.steering) * (180.0 / 60.0)
        self.steering = np.clip(self.steering,-180.0, 180.0)
        self.tire_angle = self.steering / 12.0

        #heading update
        self.heading += (self.velocity / self.wheel_base) * np.tan(np.deg2rad(self.tire_angle)) * dt

        #throttle and brake update
        if self.throttle > 0.0 and self.brake > 0.0:
            self.throttle = 0.0
      
        self.throttle += self.throttle_force * (2.0 / 60.0) - (1.0 / 60.0)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.brake += self.brake_force * (2.0 / 60.0) - (1.0 / 60.0)
        self.brake = np.clip(self.brake, 0.0, 1.0)


        #experimental physics update
        max_accel = 5.0 # m/s^2
        max_decel = 10.0 # m/s^2
        max_velocity = 40.0 # m/s

        self.velocity += self.throttle * max_accel * dt
        self.velocity -= self.brake * max_decel * dt
        self.velocity = np.clip(self.velocity, 0.0, max_velocity)

        #update position
        self.position[0] += self.velocity * np.cos(self.heading) * dt
        self.position[1] += self.velocity * np.sin(self.heading) * dt
        return

