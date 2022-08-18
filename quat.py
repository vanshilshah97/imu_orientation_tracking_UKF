import numpy as np

class Quat:
    def __init__(self):
        pass

    def multiply(self,q1,q2):
        prod = np.ones((4,q2.shape[1]))
        # prod1 = np.ones((4,1))
        # prod[0,:] = q1[0, :] * q2[0, :] - np.dot(q1[1:].T, q2[1:])[0, :]
        # prod[1:] = q1[0, :] * q2[1:] + q2[0, :] * q1[1:] + np.cross(q1[1:].T, q2[1:].T).T

        prod[0, :] = q1[0, :] * q2[0, :]-q1[1, :] * q2[1, :]-q1[2, :] * q2[2, :]-q1[3, :] * q2[3, :]
        prod[1, :] =  q1[0, :] * q2[1, :]+q1[1, :] * q2[0, :]+q1[2, :] * q2[3, :]-q1[3, :] * q2[2, :]
        prod[2, :] = q1[0, :] * q2[2, :] - q1[1, :] * q2[3, :] + q1[2, :] * q2[0, :] + q1[3, :] * q2[1, :]
        prod[3, :] = q1[0, :] * q2[3, :] + q1[1, :] * q2[2, :] - q1[2, :] * q2[1, :] + q1[3, :] * q2[0, :]

        return prod

    def inverse(self,q1):
        q_inv = np.zeros(q1.shape)
        q_inv[0,:] = q1[0,:]
        q_inv[1:] = -q1[1:]
        mag = np.linalg.norm(q1,axis=0)**2
        q_inv = q_inv/mag
        return q_inv

    def mean(self,X):
        q_est = np.zeros((4,1))
        q_est[0,0]=1
        rot_e_sum = 100
        while(rot_e_sum >0.01):
            e=np.zeros(X.shape)
            rot_e=np.zeros((3,X.shape[1]))
            for i in range(X.shape[1]):
                e[:,[i]] = self.multiply(X[:,[i]], self.inverse(q_est))
            for i in range(e.shape[1]):
                rot_e[:,[i]] = self.quat_to_rotvec(e[:,[i]])
            rot_e_mean =  np.mean(rot_e,axis=1).reshape(-1,1)
            rot_e_sum = np.linalg.norm(rot_e_mean,axis=0)
            quat_e_mean = self.rotvec_to_quat(rot_e_mean)
            q_est = self.multiply(quat_e_mean, q_est)
        return q_est,rot_e

    def g_to_quat(self,g):
        g_quat= np.pad(g,[(1,0),(0,0)],mode='constant')
        return g_quat

    def quat_to_g(self,g_quat):
        g = g_quat[1,:]
        return g

    def rotvec_to_quat(self,rotvec):
        mag = np.linalg.norm(rotvec,axis=0)
        mag[mag==0]=0.0001
        quat = np.ones((4,rotvec.shape[1]))
        quat[0,:] = np.cos(mag/2)
        quat[1:] = (rotvec/mag)*np.sin(mag/2)
        return quat


    def quat_to_rotvec(self,quat):
        quat = quat/np.linalg.norm(quat)

        rotvec = quat[1:]
        mag = 2*np.arccos(quat[0,0])
        mag = mag/np.sin(mag/2)
        rotvec = mag*rotvec
        return rotvec

    def euler_angles(self,q):
        r = np.arctan2(2 * (q[0,0] * q[1,0] + q[2,0] * q[3,0]), \
                       1 - 2 * (q[1,0] ** 2 + q[2,0] ** 2))
        p = np.arcsin(2 * (q[0,0] * q[2,0] - q[3,0] * q[1,0]))
        y = np.arctan2(2 * (q[0,0] * q[3,0] + q[1,0] * q[2,0]), \
                       1 - 2 * (q[2,0] ** 2 + q[3,0] ** 2))
        return np.array([[r], [p], [y]])

if __name__ == "__main__":
    q=Quat()
    a = np.array([[3,3],[1,1],[-2,-2],[1,1]])
    b = np.array([[2,2], [-1,-1], [2,2], [3,3]])
    X = np.array([
        [0.92707,0.90361,0.75868],[0.02149,0.0025836,-0.21289],[0.19191,0.097279,0.53263],[0.32132,0.41716,0.30884]])
    # X = np.array([[0.92707, 0.90361], [0.02149, 0.0025836], [0.19191, 0.097279],[0.32132, 0.41716]])
    mean = q.mean(X)
    # prod = q.multiply(a,b)
    # quat=np.array([[0.53767],[1.8339],[-2.2588],[0.86217]])
    # rot_vec = q.quat_to_rotvec(quat)
    # quat_vec = q.rotvec_to_quat(rot_vec)
    # print (mean)
    prod = q.multiply(a,b)
    inv = q.inverse(a)
    print (inv)
    # ans = q.rotvec_to_quat(np.array([[0],[np.pi/2],[0]]))
    # ans = q.inverse(np.array([[0],[0],[0],[9.8]]))
    # print (mean)

