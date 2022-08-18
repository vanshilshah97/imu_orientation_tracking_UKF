#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an extended kalman filter
import numpy as np
from scipy import io
import os
import matplotlib.pyplot as plt

def quat_multiply(q1, q2):
    if(q2.shape[1]>q1.shape[1]):
        shape = q2.shape
    else:
        shape = q1.shape
    prod = np.ones(shape)
    prod[0, :] = q1[0, :] * q2[0, :] - q1[1, :] * q2[1, :] - q1[2, :] * q2[2, :] - q1[3, :] * q2[3, :]
    prod[1, :] = q1[0, :] * q2[1, :] + q1[1, :] * q2[0, :] + q1[2, :] * q2[3, :] - q1[3, :] * q2[2, :]
    prod[2, :] = q1[0, :] * q2[2, :] - q1[1, :] * q2[3, :] + q1[2, :] * q2[0, :] + q1[3, :] * q2[1, :]
    prod[3, :] = q1[0, :] * q2[3, :] + q1[1, :] * q2[2, :] - q1[2, :] * q2[1, :] + q1[3, :] * q2[0, :]
    return prod
def quat_inverse(q1):
    q_inv = np.zeros(q1.shape)
    q_inv[0, :] = q1[0, :]
    q_inv[1:] = -q1[1:]
    mag = np.linalg.norm(q1, axis=0) ** 2
    q_inv = q_inv / mag
    return q_inv

def quat_to_rotvec(quat):
    quat = quat / np.linalg.norm(quat)
    rotvec = quat[1:]
    mag = 2 * np.arccos(quat[0, 0])
    mag = mag / np.sin(mag / 2)
    rotvec = mag * rotvec
    return rotvec

def rotvec_to_quat(rotvec):
    mag = np.linalg.norm(rotvec,axis=0)
    mag[mag==0]=0.0001
    quat = np.ones((4,rotvec.shape[1]))
    quat[0,:] = np.cos(mag/2)
    quat[1:] = (rotvec/mag)*np.sin(mag/2)
    return quat

def qua_mean( X):
    q_est = np.zeros((4, 1))
    q_est[0, 0] = 1
    rot_e_sum = 100
    e = np.zeros((4, 13))
    rot_e = np.zeros((3, 13))
    while (rot_e_sum > 0.9):

        for i in range(13):
            e[:, [i]] = quat_multiply(X[:, [i]], quat_inverse(q_est))
        for i in range(e.shape[1]):
            rot_e[:, [i]] = quat_to_rotvec(e[:, [i]])
        rot_e_mean = np.mean(rot_e, axis=1).reshape(-1, 1)
        rot_e_sum = np.linalg.norm(rot_e_mean, axis=0)
        quat_e_mean = rotvec_to_quat(rot_e_mean)
        q_est = quat_multiply(quat_e_mean, q_est)

    return q_est, rot_e

def euler_angles(q):
    r = np.arctan2(2 * (q[0, 0] * q[1, 0] + q[2, 0] * q[3, 0]), \
                   1 - 2 * (q[1, 0] ** 2 + q[2, 0] ** 2))
    p = np.arcsin(2 * (q[0, 0] * q[2, 0] - q[3, 0] * q[1, 0]))
    y = np.arctan2(2 * (q[0, 0] * q[3, 0] + q[1, 0] * q[2, 0]), \
                   1 - 2 * (q[2, 0] ** 2 + q[3, 0] ** 2))
    return np.array([[r], [p], [y]])


def data_loader(imu_data):
    accel_data = imu_data['vals'][:3, :]
    ax_data = accel_data[0, :]
    ay_data = accel_data[1, :]
    az_data = accel_data[2, :]
    Vref = 3300
    bias_x = 510
    bias_y = 501
    bias_z = 502
    sen = 34.2
    scale = Vref / (1023 * sen)
    ax_data = (-1*(ax_data - bias_x * (np.ones(ax_data.shape))) * scale).reshape(1,-1)
    ay_data = (-1*(ay_data - bias_y * (np.ones(ay_data.shape))) * scale).reshape(1,-1)
    az_data = ((az_data - bias_z * (np.ones(az_data.shape))) * scale).reshape(1,-1)
    accel_data = np.zeros((3,ax_data.shape[1]))
    accel_data[0,:] = ax_data
    accel_data[1, :] = ay_data
    accel_data[2, :] = az_data

    gyro_data = imu_data['vals'][3:6, :]

    sen_roll = 390
    sen_pitch = 195
    sen_yaw = 200

    gy_bias_x = 373.7
    gy_bias_y = 375.8
    gy_bias_z = 369.8

    wx_data = gyro_data[1, :]
    wy_data = gyro_data[2, :]
    wz_data = gyro_data[0, :]
    scale_x = (np.pi/180)*(Vref / (1023 * sen_roll))
    scale_y = (np.pi/180)*(Vref / (1023 * sen_pitch))
    scale_z = (np.pi/180)*(Vref / (1023 * sen_yaw))

    wx_data = ((wx_data - gy_bias_x * (np.ones(wx_data.shape))) * scale_x).reshape(1,-1)
    wy_data = ((wy_data - gy_bias_y * (np.ones(wy_data.shape))) * scale_y).reshape(1,-1)
    wz_data = ((wz_data - gy_bias_z * (np.ones(wz_data.shape))) * scale_z).reshape(1,-1)

    gyro_data = np.zeros((3, ax_data.shape[1]))
    gyro_data[0, :] = wx_data
    gyro_data[1, :] = wy_data
    gyro_data[2, :] = wz_data

    ts = imu_data['ts']
    return accel_data,gyro_data,ts

def comp_sigma(x_prevk,cov_prevk,Q,W,X):
    # print(cov_prevk+Q)

    S = 3.4641 * np.linalg.cholesky(cov_prevk+Q)
    W[:,:6] = S
    W[:, 6:] = -S

    quat_w = rotvec_to_quat(W[:3,:])
    quat = quat_multiply(x_prevk[:4],quat_w)
    X[:4,:-1] = quat
    X[4:,:-1] = x_prevk[4:]+ W[3:,:]
    X[:, [-1]] = x_prevk

    return X

def transform_sigma(X,dt,q,cov_w):

    mag = np.linalg.norm(X[4:,:],axis=0)
    mag[mag==0]=0.0001
    vect = X[4:, :] / mag
    q[0,:] = np.cos(mag * (dt / 2))
    q[1:] = np.sin(mag * (dt / 2)) * vect

    X[:4, :] = quat_multiply(X[:4,:], q)

    x_mean_k,Pk,cov_w = comp_xk_Pk(X,cov_w)
    return X,x_mean_k,Pk,cov_w
def comp_xk_Pk(Y,cov_w):
    quat = Y[:4,:]
    quat_mean,quat_e = qua_mean(quat)
    w_mean_k = np.mean(Y[4:,:],axis=1).reshape((-1,1))
    x_mean_k = np.vstack((quat_mean,w_mean_k))

    cov_w[:3,:] = quat_e
    cov_w[3:,:] = Y[4:,:]-x_mean_k[4:]
    Pk = np.matmul(cov_w,cov_w.T)/13
    return x_mean_k,Pk,cov_w

def predict_z(Y,cov_w,R,Z,g_quat):

    res = quat_multiply(g_quat, Y[:4,:])
    inv = quat_inverse(Y[:4, :])

    Z[:3, :] = quat_multiply(inv, res)[1:]
    Z[3:, :] = Y[4:, :]

    z_mean = np.mean(Z,axis=1).reshape(-1,1)

    w_z = Z-z_mean
    Pzz = np.matmul(w_z,w_z.T)/13

    Pvv = Pzz+R
    Pxz = np.matmul(cov_w,w_z.T)/13
    return Pvv,Pxz,z_mean

def kalman_update(Pk,Pvv,Pxz,z_mean,accel_data,gyro_data,x_mean_k,z_mes,gain_up):
    K = np.matmul(Pxz,np.linalg.inv(Pvv))
    z_mes[:3,[0]] = accel_data
    z_mes[3:, [0]] = gyro_data
    inov = z_mes-z_mean
    gain = np.matmul(K,inov)

    gain_up[:4] = rotvec_to_quat(gain[:3])
    gain_up[4:] = gain[3:]
    x_mean_k[:4] = quat_multiply(x_mean_k[:4],gain_up[:4])
    x_mean_k[4:] = x_mean_k[4:]+gain_up[4:]
    Pk = Pk-np.matmul(np.matmul(K,Pvv),K.T)
    return x_mean_k,Pk

def estimate_rot(number):
    #your code goes here
    filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(number) + ".mat")
    imu_data = io.loadmat(filename)
    accel_data, gyro_data, ts = data_loader(imu_data)
    x, y, z = accel_data[:, 0]
    roll = np.arctan2(y, z)
    pitch = np.arctan2(-x, np.sqrt(y ** 2 + z ** 2))
    yaw = 0
    size = accel_data.shape[1]
    pred_roll = np.zeros(size)
    pred_pitch = np.zeros(size)
    pred_yaw = np.zeros(size)
    pred_roll[0] = roll
    pred_pitch[0] = pitch
    pred_yaw[0] = yaw

    x_prev_k = np.zeros((7, 1))
    x_prev_k[:4] = rotvec_to_quat(np.array([[roll], [pitch], [yaw]]))
    x_prev_k[4:] = gyro_data[:, 0].reshape(-1, 1)

    cov_prev_k = 50 * np.array([[1., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0.],
                                [0., 0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0., 0.],
                                [0., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 0., 1]])
    Q = 11 * np.array([[1, 0., 0., 0., 0., 0.],
                       [0., 1, 0., 0., 0., 0.],
                       [0., 0., 1, 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 1, 0.],
                       [0., 0., 0., 0., 0., 1]])
    R = 15 * np.array([[1, 0., 0., 0., 0., 0.],
                       [0., 1, 0., 0., 0., 0.],
                       [0., 0., 1, 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 1, 0.],
                       [0., 0., 0., 0., 0., 1]])
    W = np.zeros((6, 12))
    X = np.zeros((7, 13))
    q = np.zeros((4, 13))
    cov_w = np.zeros((6, 13))
    g_quat = np.array([[0], [0], [0], [9.8]])
    Z = np.zeros((6, 13))
    z_mes = np.zeros((6, 1))
    gain_up = np.zeros((7, 1))

    for i in range(1, size):

        sigma = comp_sigma(x_prev_k, cov_prev_k,Q,W,X)
        Y, x_mean_k, Pk, cov_w = transform_sigma(sigma, ts[0, i] - ts[0, i - 1],q,cov_w)
        Pvv, Pxz, z_mean = predict_z(Y, cov_w,R,Z,g_quat)
        x_prev_k, cov_prev_k = kalman_update(Pk, Pvv, Pxz, z_mean, accel_data[:, [i]],
                                             gyro_data[:, [i]], x_mean_k,z_mes,gain_up)
        rot = euler_angles(x_prev_k[:4])

        pred_roll[i], pred_pitch[i], pred_yaw[i] = rot[0, 0], rot[1, 0], rot[2, 0]

    vicon_filename = os.path.join(os.path.dirname(__file__), "vicon/viconRot" + str(number) + ".mat")
    vicon_data = io.loadmat(vicon_filename)
    vicon_data_rots = vicon_data['rots']
    vicon_roll = np.arctan2(vicon_data_rots[2, 1, :], vicon_data_rots[2, 2, :])
    vicon_pitch = np.arctan2(-vicon_data_rots[2, 0,:],np.sqrt(vicon_data_rots[2, 1, :] ** 2 + vicon_data_rots[2, 2, :] ** 2))
    vicon_yaw = np.arctan2(vicon_data_rots[1,0,:],vicon_data_rots[0,0,:])

    plt.figure(1)
    plt.plot(vicon_roll,'k')
    plt.plot(pred_roll)
    plt.figure(2)
    plt.plot(vicon_pitch,'k')
    plt.plot(pred_pitch)
    plt.figure(3)
    plt.plot(vicon_yaw, 'k')
    plt.plot(pred_yaw)
    plt.show()

estimate_rot(2)