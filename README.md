# Python Inertial Measurement Unit Simulator

Simulates the accelerometer and gyroscope readings that an IMU would produce which is following a trajectory in 3D space. Trajectories can be loaded from ROS1 bagfiles, or simulated trajectories can be used. 

## Depends

- numpy
- scipy
- matplotlib
- rosbag _(optional)_

## How to use

Have a look at 'demo_simulate_imu.py'. 
Install dependencies with pip.
If you can not install rosbag with pip directly, try installing 'bagpy'.
Somehow this will install rosbag correctly.