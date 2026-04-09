import numpy as np

class Car:
    def __init__(self, position, heading=0.0, velocity=0.0):
        self.position = np.array(position, dtype=float)
        self.heading = heading
        self.velocity = velocity
        self.width = 1.9
        self.length = 3.9
        

    def get_corners(self):
        cx, cy = self.position
        L = self.length / 2
        W = self.width / 2

        # local rectangle corners
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