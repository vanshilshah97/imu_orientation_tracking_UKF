# Unscented Kalman Filter for orientation tracking

The ground truth of the orientation is captured by the vicon and is present in the `vicon` folder. The IMU datasets is present in the `imu` folder. estimate_rot.py tracks the orientation using the IMU data and UKF.

## Dataset 1
<img src="https://github.com/Nagarakshith1/UKF_orientation_tracking/blob/master/images/1_pitch.png?raw=true" width="300" height="300"> <img src="https://github.com/Nagarakshith1/UKF_orientation_tracking/blob/master/images/1_roll.jpeg?raw=true" width="300" height="300"> <img src="https://github.com/Nagarakshith1/UKF_orientation_tracking/blob/master/images/1_yaw.jpeg?raw=true" width="300" height="300">

## Dataset 2
<img src="https://github.com/Nagarakshith1/UKF_orientation_tracking/blob/master/images/2_pitch.jpeg?raw=true" width="300" height="300"> <img src="https://github.com/Nagarakshith1/UKF_orientation_tracking/blob/master/images/2_roll.jpeg?raw=true" width="300" height="300"> <img src="https://github.com/Nagarakshith1/UKF_orientation_tracking/blob/master/images/2_yaw.jpeg?raw=true" width="300" height="300">

## Reference
> E. Kraft, "A quaternion-based unscented Kalman filter for orientation tracking," Sixth International Conference of Information Fusion, 2003. Proceedings of the, Cairns, Queensland, Australia, 2003, pp. 47-54, doi: 10.1109/ICIF.2003.177425.
