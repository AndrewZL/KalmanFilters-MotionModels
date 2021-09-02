import numpy as np
from KalmanFilter import KF

class SixDOF_Tracker:
    def __init__(self, x0, P0, timestep=0.1):
        assert(len(x0) == 18)

        self.dt = timestep

        self.A_s = np.matrix([
            [1, 0, 0, self.dt, 0, 0, 0.5 * self.dt * self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0, 0, 0.5 * self.dt * self.dt, 0],
            [0, 0, 1, 0, 0, self.dt, 0, 0, 0.5 * self.dt * self.dt],
            [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.A = np.block([
            [self.A_s, np.zeros((self.A_s.shape[0], self.A_s.shape[0]))],
            [np.zeros((self.A_s.shape[0], self.A_s.shape[0])), self.A_s]
        ])
        self.H = np.vstack((
            np.concatenate((np.identity(3), np.zeros((3, 15))), axis=1),
            np.concatenate((np.zeros((3, 9)), np.identity(3), np.zeros((3, 6))), axis=1)))

        self.G = np.matrix(
            [0.5 * self.dt ** 2, 0.5 * self.dt ** 2, 0.5 * self.dt ** 2,
             self.dt, self.dt, self.dt,
             1, 1, 1]).T
        self.Q_s = self.G * self.G.T * self.sigma_a ** 2
        self._Q = np.block([
            [self.Q_s, np.zeros((self.Q_s.shape[0], self.Q_s.shape[0]))],
            [np.zeros((self.Q_s.shape[0], self.Q_s.shape[0])), self.Q_s]])

        self.kf = KF(dim_x=18, dim_z=6, dt=self.dt, A=self.A, H=self.H, x0=x0, P0=P0)


