import numpy as np


class ExtendedKalmanFilter:

    def __init__(self, x, P):

        self.x = x
        self.P = P

    def propagate(self, u, dt, Q):

        # propagate state x
        x, y, theta = self.x
        v, omega = u
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.x += np.array([dx, dy, dtheta])

        # propagate covariance P
        G = np.array([
            [1., 0., - r * np.cos(theta) + r * np.cos(theta + dtheta)],
            [0., 1., - r * np.sin(theta) + r * np.sin(theta + dtheta)],
            [0., 0., 1.]
        ])  # Jacobian of state transition function

        self.P = G @ self.P @ G.T + Q

    def update(self, z, R):

        # compute Kalman gain
        H = np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])  # Jacobian of observation function
        H = H.astype(np.float)

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)

        # update state x
        x, y, theta = self.x
        z_ = np.array([x, y])  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)

        # update covariance P
        self.P = self.P - K @ H @ self.P


