import numpy as np
from scipy import io
import matplotlib.pyplot as plt

def rpy_Accel(imu_data):
    Vref = 3300
    bias_x=510
    bias_y=501
    bias_z=502
    sen = 34.2
    scale = Vref/(1023*sen)
    print (scale)
    accel_data = imu_data[:3,:]

    ax_data = accel_data[0,:]
    ay_data = accel_data[1, :]
    az_data = accel_data[2, :]

    ax_data = (ax_data-bias_x*(np.ones(ax_data.shape)))*scale
    ay_data = (ay_data-bias_y*(np.ones(ay_data.shape)))*scale
    az_data = (az_data-bias_z*(np.ones(az_data.shape)))*scale

    # roll = np.arctan2(-accel_data[1,:],accel_data[2,:])
    # pitch = np.arctan2(accel_data[0,:],np.sqrt(accel_data[1,:]**2+accel_data[2,:]**2))

    plt.show()
    print(az_data)
    roll = np.arctan2(-ay_data, az_data)
    pitch = np.arctan2(ax_data,np.sqrt(ay_data**2+az_data**2))
    print(roll)

    return roll, pitch

if __name__ == "__main__":
    imu_data = io.loadmat('imu/imuRaw1.mat')
    

    vicon_data = io.loadmat('vicon/viconRot1.mat')
    vicon_data_rots = vicon_data['rots']

    accel_roll, accel_pitch = rpy_Accel(imu_data['vals'])
    vicon_roll = np.arctan2(vicon_data_rots[2,1,:],vicon_data_rots[2,2,:])
    vicon_pitch = np.arctan2(-vicon_data_rots[2,0],np.sqrt(vicon_data_rots[2,1,:]**2+vicon_data_rots[2,2,:]**2))

    plt.figure(1)
    plt.plot(vicon_roll,'k')
    plt.plot(accel_roll)
    plt.figure(2)
    plt.plot(vicon_pitch,'k')
    plt.plot(accel_pitch)
    plt.show()
