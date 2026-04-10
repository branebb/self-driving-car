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