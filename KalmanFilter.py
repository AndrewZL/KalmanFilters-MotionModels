import numpy as np


class KF:
    """
    6 DOF Linear Kalman Filter for position + orientation tracking
    A: System dynamics matrix
    Q: Process noise covariance
    R: Measurement noise covariance
    P: Estimate error covariance
    """
    def __init__(self, dim_x, dim_z, dt, A, H, x0, P0, Q=None, R=None):
        # Dimensions
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dt = dt
        # State transition matrices
        self.A = A
        self.H = H
        # Initialization
        self.P = P0
        self.x = x0
        # Noise covariance matrices
        if isinstance(Q, list):
            self._Q = Q
        else:
            self.Q = np.eye(dim_x) * 1e-10
        if isinstance(R, list):
            self._R = R
        else:
            self._R = np.matrix(self._R * np.identity(self.H.shape[0]))

    def update(self, z):
        x_hat = self.A * self.x
        P_hat = self.A * self.P * self.A.T + self._Q
        K = P_hat * self.H.T * np.linalg.pinv(self.H * P_hat * self.H.T + self._R)

        self.x = x_hat + K * (z - self.H * x_hat)
        self.P = (np.identity(self.P.shape[0]) - K * self.H) * P_hat
