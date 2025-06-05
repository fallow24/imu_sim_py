# Python Inertial Measurement Unit Simulator

Simulates the accelerometer and gyroscope readings that an IMU would produce which is following a trajectory in 3D space. Trajectories can be loaded from ROS1 bagfiles, or simulated trajectories can be used.  

![screenshot](https://github.com/fallow24/imu_sim_py/blob/main/img/screenshot.png?raw=true)

## Depends

- numpy
- scipy
- matplotlib
- rosbag _(optional)_

Install dependencies with pip.
If you can not install rosbag with pip directly, try installing `bagpy`.
Somehow this will install rosbag correctly.

## How to use

Have a look at `demo_simulate_imu.py`. 
Alternatively, here is a minimal example:

```python
import numpy as np
import matplotlib.pyplot as plt
import trajectories as trj
import accelerometer as accel
import gyroscope as gyro

# Create 10 seconds of a non changing pose, sampled at 200Hz
poses = trj.generate_stationary_pose(duration=10, fs=200)

# What the IMU would measure for the specified trajectory
acc = accel.readings(poses)
gyr = gyro.readings(poses)
gyr = np.rad2deg(gyr)  # convert to deg/s

# Plot the accelerometer data using timestamp as x axis
timestamps = acc[:, 0] # same stamps as gyr and poses
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# Accelerometer plot
g = 9.80665
axs[0].plot(timestamps, acc[:, 1]/g, label='Accel X')
axs[0].plot(timestamps, acc[:, 2]/g, label='Accel Y')
axs[0].plot(timestamps, acc[:, 3]/g, label='Accel Z')
axs[0].set_ylabel('Acceleration (g)')
axs[0].set_title('Simulated Accelerometer Readings (Gravity + Movement)')
axs[0].legend()
axs[0].grid(True)
# Gyroscope plot
axs[1].plot(timestamps, gyr[:, 1], label='Gyro X')
axs[1].plot(timestamps, gyr[:, 2], label='Gyro Y')
axs[1].plot(timestamps, gyr[:, 3], label='Gyro Z')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Angular Rate (deg/s)')
axs[1].set_title('Simulated Gyroscope Readings')
axs[1].legend()
axs[1].grid(True)
# Show plot
plt.tight_layout()
plt.show()
```

## IMU bias + noise model

The bias and noise models are implemented according to [kalibr](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model). You can define accelerometer and gyroscope sensor random walk and noise densities.
If you do not explicitly state any bias, random walk, or noise density in the `readings(...)` function (as in the example given above), the program will assume that there is no bias (all elements are zero). 
Random walk and noise densities will default to behaving like a MPU-6050 sensor (e.g., Gyro Noise Density = 0.005 rad/s/√Hz, Gyro Random Walk = 0.0002 rad/s²/√Hz, Accel Noise Density = 0.00098 m/s²/√Hz, and Accel Random Walk = 0.00002 m/s³/√Hz)
You can obtain the "perfect" sensor by setting all these values to zero, e.g.:
```python
acc = accel.readings(poses, random_walk=0, noise_density=0)
gyr = gyro.readings(poses, random_walk=0, noise_density=0)
```
		 
