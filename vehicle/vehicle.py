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
    def __init__(self, position, heading, velocity):
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
        self.sterring = 0.0 #current steering, from -180 to 180 degrees, updated outside of this class
        

    def update(self, velocity):
        self.velocity = velocity
        



