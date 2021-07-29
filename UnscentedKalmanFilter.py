import numpy as np
import scipy.linalg


class UKF:
    """
    Straightforward python implementation of Unscented Kalman Filter
    [1] E. A. Wan and R. Van Der Merwe, “The unscented Kalman filter for nonlinear estimation,”
        in Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium,
        Oct. 2000
    [2] R. Schubert, E. Richter, and G. Wanielik,
        “Comparison and evaluation of advanced motion models for vehicle tracking,”
        in 2008 11th International Conference on Information Fusion, Jun. 2008
    """
    def __init__(self, dim_x, dim_z, dt, fx, hx, alpha, beta, kappa, P0, x0):
        """
        Unscented Kalman Filter Constructor
        :param dim_x: dimension of state vector
        :param dim_z: dimension of measurement vector
        :param dt: timestep
        :param fx: state transition function
        :param hx: measurement function (state -> measurement)
        :param alpha: spread of sigma points around mean (typically small, ~1e-3)
        :param beta: prior knowledge of distribution of state (2 for perfect Gaussian)
        :param kappa: secondary scaling parameter (= 3 - L)
        :param P0: initial estimate for covariance matrix (dim_x, dim_x)
        :param x0: initial estimate for system state (dim_x)
        """
        # Dimensions
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dt = dt
        # Functions
        self.fx = fx
        self.hx = hx
        # Initialization
        self.x = x0  # np.zeros(dim_x)
        self.P = P0  # np.eye(dim_x)
        # Process noise covariance matrix (Large Q -> Tracks larger changes more closely)
        self.Q = np.eye(dim_x) * 1e-10
        # Measurement noise covariance matrix (Small R -> Tracks measurements closely)
        self.R = np.eye(dim_z) * 0
        # Sigma points parameters
        self.n_sigma = 2 * dim_x + 1
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        # Scaling parameters
        self._lambda = np.power(alpha, 2) * (self._dim_x + kappa) - self._dim_x
        # Initialize sigma points
        self.sigma_points = None
        self.mean_weights, self.covariance_weights = self.get_weights()
        self.sigma_x = None
        self.sigma_z = None

    def get_sigma_points(self):
        """
        Calculates the sigma points by method used in [1]
        :return: sigma points
        """
        U = scipy.linalg.cholesky((self._lambda + self._dim_x) * self.P)
        sigma_points = np.zeros((self.n_sigma, self._dim_x))
        sigma_points[0] = self.x
        for i in range(self._dim_x):
            sigma_points[i+1] = self.x + U[i]
            sigma_points[self._dim_x+i+1] = self.x - U[i]
        return sigma_points

    def get_weights(self):
        """
        Calculates the corresponding mean and covariance weight of each sigma point vector
        :return: mean and covariance weights
        """
        W_mean = np.full(self.n_sigma, 1 / (2 * (self._dim_x + self._lambda)))
        W_covariance = np.full(self.n_sigma, 1 / (2 * (self._dim_x + self._lambda)))
        W_mean[0] = self._lambda / (self._dim_x + self._lambda)
        W_covariance[0] = W_mean[0] + (1 - self._alpha * self._alpha + self._beta)
        return W_mean, W_covariance

    def predict(self):
        """
        Performs prediction step (system propagation)
        """
        # Transforms sigma points to next state
        self.sigma_points = self.get_sigma_points()
        self.sigma_x = np.atleast_2d(np.array([self.fx(s) for s in self.sigma_points]))
        # Unscented Transform
        # Compute state mean (mean weights * transformed sigma points)
        self.x = np.dot(self.mean_weights, self.sigma_x)
        # Compute Noise Covariance Matrix
        residual_x = self.sigma_x - self.x[np.newaxis, :]
        self.P = np.dot(residual_x.T, np.dot(np.diag(self.covariance_weights), residual_x))
        self.P += self.Q

    def update(self, z):
        """
        Performs update step (measurement update)
        :param z: measurement vector
        """
        # Transforms sigma points to measurement
        self.sigma_z = np.atleast_2d(np.array([self.hx(x) for x in self.sigma_x]))
        y = np.dot(self.mean_weights, self.sigma_z)
        # Compute measurement mean covariance
        residual_y = self.sigma_z - y[np.newaxis, :]
        residual_x = self.sigma_x - self.x[np.newaxis, :]
        P_yy = np.dot(residual_y.T, np.dot(np.diag(self.covariance_weights), residual_y))
        P_yy += self.R
        # Compute cross variance matrix
        P_xy = np.dot(residual_y.T, np.dot(np.diag(self.covariance_weights), residual_x)).T
        # Compute Kalman gain
        K = np.dot(P_xy, np.linalg.inv(P_yy))
        # Update state
        self.x += np.dot(K, (z - y))
        print(self.x)
        self.P -= np.dot(K, np.dot(P_yy, K.T))
