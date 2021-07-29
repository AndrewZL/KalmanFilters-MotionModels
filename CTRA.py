import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from UnscentedKalmanFilter import UKF


class CTRA_Tracker:
    def __init__(self, x0, P0, timestep=0.1, alpha=1e-2, beta=2, kappa=-3):
        self.dt = timestep
        self.kf = UKF(dim_x=6, dim_z=3, dt=timestep, fx=self.f, hx=self.h,
                      alpha=alpha, beta=beta, kappa=kappa, x0=x0, P0=P0)

    def f(self, x):
        """
        State transition function for CTRA
        :param x: state vector
        :return: updated state vector
        """
        s_x, s_y, theta, v, a, omega = x
        s_x += 1 / np.power(omega, 2) * (
                (v * omega + a * omega * self.dt) * np.sin(theta + omega * self.dt) +
                a * np.cos(theta + omega * self.dt) - v * omega * np.sin(theta) - a * np.cos(theta))
        s_y += 1 / np.power(omega, 2) * (
                (-v * omega - a * omega * self.dt) * np.cos(theta + omega * self.dt) +
                a * np.sin(theta + omega * self.dt) + v * omega * np.cos(theta) - a * np.sin(theta))
        theta += omega * self.dt
        v += a * self.dt
        a += 0
        omega += 0
        x = np.array([s_x, s_y, theta, v, a, omega])
        return x

    def h(self, x):
        """
        Measurement function for CTRA
        :param x: state vector
        :return: array containing x, y, and theta (yaw)
        """
        return np.array([x[0], x[1], x[2]])


if __name__ == '__main__':
    # load sample data
    measurements = pd.read_csv('measurements.csv').to_numpy()
    x_0 = np.array([-4.49969901e+00, -4.23095983e+01,  -4.29246303e-02, 1, -1e-3, -1e-3])
    P_0 = np.eye(6) * 0.2
    tracker = CTRA_Tracker(x_0, P_0)

    # run tracker on sample data
    kfr = []
    for i in range(len(measurements)):
        tracker.kf.predict()
        tracker.kf.update(measurements[i])
        kfr.append(tracker.kf.x)

    # visualization
    gt_df = pd.DataFrame(measurements)
    kf_df = pd.DataFrame(kfr)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title('X')
    ax1.scatter(range(len(gt_df)), gt_df[0])
    ax1.plot(kf_df[0], color='orange')
    ax2.set_title('Y')
    ax2.scatter(range(len(gt_df)), gt_df[1])
    ax2.plot(kf_df[1], color='orange')
    ax3.set_title('Yaw')
    ax3.scatter(range(len(gt_df)), gt_df[2])
    ax3.plot(kf_df[2], color='orange')
    plt.legend(['Ground Truth', 'CTRA'], loc='lower right')
    plt.show()
