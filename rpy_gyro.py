import numpy as np
from scipy import io
import matplotlib.pyplot as plt

def rpy_gyro(imu_data,ts):
    gyro_data = imu_data[3:6,:]

    Vref  = 3300
    sen_roll = 210
    sen_pitch = 195
    sen_yaw = 200
    scale = Vref / (1023 * sen_pitch)
    # bias_x = np.sum(gyro_data[1,:5])/5
    bias_x = 373.7
    # print(bias_x)
    # bias_y = np.sum(gyro_data[2,:5])/5
    bias_y = 375.8
    # print (bias_y)
    # bias_z = np.sum(gyro_data[0,:5])/5
    bias_z = 369.8
    print(bias_z)

    wx_data = gyro_data[1, :]
    wy_data = gyro_data[2, :]
    wz_data = gyro_data[0, :]

    accel_data = imu_data[:3, :]

    ax_data = accel_data[0, 0]
    ay_data = accel_data[1, 0]
    az_data = accel_data[2, 0]

    ax_data = (ax_data - 510 * (np.ones(ax_data.shape))) * scale
    ay_data = (ay_data - 501 * (np.ones(ay_data.shape))) * scale
    az_data = (az_data - 502 * (np.ones(az_data.shape))) * scale

    wx_data = (wx_data - bias_x * (np.ones(wx_data.shape))) * scale

    wy_data = (wy_data - bias_y * (np.ones(wy_data.shape))) * scale
    wz_data = (wz_data - bias_z * (np.ones(wz_data.shape))) * scale

    roll = np.ones(ts.shape[1])
    roll[0] = np.arctan2(-ay_data, az_data)
    pitch = np.ones(ts.shape[1])
    pitch[0] = np.arctan2(ax_data,np.sqrt(ay_data**2+az_data**2))
    yaw = np.ones(ts.shape[1])
    yaw[0]=0
    for i in range(1,ts.shape[1]):

        roll[i] = roll[i-1]+wx_data[i-1]*(ts[0,i]-ts[0,i-1])
        pitch[i] = pitch[i - 1] + wy_data[i - 1] * (ts[0, i] - ts[0, i - 1])
        yaw[i] = yaw[i - 1] + wz_data[i - 1] * (ts[0, i] - ts[0, i - 1])
    return roll,pitch,yaw

if __name__ == "__main__":
    imu_data = io.loadmat('imu/imuRaw1.mat')
    vicon_data = io.loadmat('vicon/viconRot1.mat')
    vicon_data_rots = vicon_data['rots']

    gyro_roll, gyro_pitch, gyro_yaw = rpy_gyro(imu_data['vals'],imu_data['ts'])
    vicon_roll = np.arctan2(vicon_data_rots[2, 1, :], vicon_data_rots[2, 2, :])
    vicon_pitch = np.arctan2(-vicon_data_rots[2, 0],np.sqrt(vicon_data_rots[2, 1, :] ** 2 + vicon_data_rots[2, 2, :] ** 2))
    vicon_yaw = np.arctan2(vicon_data_rots[1, 0],vicon_data_rots[0, 0])
    plt.figure(1)
    plt.plot(vicon_roll, 'k')
    plt.plot(gyro_roll)

    plt.figure(2)
    plt.plot(vicon_pitch, 'k')
    plt.plot(gyro_pitch)

    plt.figure(3)
    plt.plot(vicon_yaw, 'k')
    plt.plot(gyro_yaw)
    plt.show()
