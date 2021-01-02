import numpy as np
import utils
import os
import gps_cali
from Extended_kalmanfilter import ExtendedKalmanFilter
import matplotlib.pyplot as plt

file_path = './gps_data/'
oxts = utils.oxts()


def parsing(path):
    data = []
    data_path = path+"data/"
    flist = os.listdir(data_path)

    for file in flist:
        with open(data_path + file) as f:
            line = f.readline()
            if not line:
                break
        line = np.array(line.split(' '))
        line = np.delete(line, np.where(line == ' '))
        oxts.get_value(line)
        f.close()

    return oxts


def gps_imu(file_path):
    oxts = parsing(file_path)
    gps_data = []

    for i in range(len(oxts.lat)):
        gps_data.append(np.array([
            oxts.lon[i],
            oxts.lat[i],
            oxts.alt[i]
        ], dtype=np.float))

    gps_data = np.array(gps_data, dtype=np.float).T

    yaws = np.array(oxts.yaw, dtype=np.float)
    yaw_rates = np.array(oxts.wz, dtype=np.float)
    forward_velocities = np.array(oxts.vf, dtype=np.float)

    origin = gps_data[:, 0]
    gps_data_trajectory = gps_cali.lla_to_enu(gps_data, origin)

    imu_x, imu_y = kalman(gps_data_trajectory, yaws, yaw_rates, forward_velocities)

    return imu_x, imu_y, gps_data_trajectory


def kalman(gps_data, yaws, yawrates, forward_velocities):
    xy_obs_noise_std = 2.0
    yaw_rate_noise_std = 0.002
    forward_velocity_noise_std = 0.003
    N = len(gps_data[0])

    initial_yaw_std = np.pi
    initial_yaw = yaws[0] + np.random.normal(0, initial_yaw_std)

    x = np.array([
        gps_data[0, 0],
        gps_data[1, 0],
        initial_yaw
    ])

    P = np.array([
        [xy_obs_noise_std ** 2., 0., 0.],
        [0., xy_obs_noise_std ** 2., 0.],
        [0., 0., initial_yaw_std ** 2.]
    ])

    Q = np.array([
        [xy_obs_noise_std ** 2., 0.],
        [0., xy_obs_noise_std ** 2.]
    ], dtype=np.float)

    R = np.array([
        [forward_velocity_noise_std ** 2., 0., 0.],
        [0., forward_velocity_noise_std ** 2., 0.],
        [0., 0., yaw_rate_noise_std ** 2.]
    ])

    # initialize Kalman filter
    kf = ExtendedKalmanFilter(x, P)

    # array to store estimated 2d pose [x, y, theta]
    mu_x = [x[0], ]
    mu_y = [x[1], ]
    mu_theta = [x[2], ]

    # array to store estimated error variance of 2d pose
    var_x = [P[0, 0], ]
    var_y = [P[1, 1], ]
    var_theta = [P[2, 2], ]

    for t_idx in range(1, N):
        dt = 0.01

        # get control input `u = [v, omega] + noise`
        u = np.array([
            forward_velocities[t_idx],
            yawrates[t_idx]
        ])

        # propagate!
        kf.propagate(u, dt, R)

        # get measurement `z = [x, y] + noise`
        z = np.array([
            gps_data[0][t_idx],
            gps_data[1][t_idx]
        ], dtype=np.float)

        # update!
        kf.update(z, Q)

        # save estimated state to analyze later
        mu_x.append(kf.x[0])
        mu_y.append(kf.x[1])
        mu_theta.append(gps_cali.normalize_angles(kf.x[2]))

    mu_x = np.array(mu_x)
    mu_y = np.array(mu_y)

    return mu_x, mu_y


if __name__ == "__main__":
    imu_x, imu_y, gps_data_trajectory = gps_imu(file_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    xs, ys, _ = gps_data_trajectory
    ax.plot(xs, ys, lw=2, label='ground-truth trajectory', color='b')
    ax.plot(imu_x, imu_y, lw=4, label='estimated trajectory', color='r')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    ax.grid()
    plt.show()
