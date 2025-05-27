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
timestamps = poses[:, 0]
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# Accelerometer plot
g = 9.80665
axs[0].plot(timestamps, acc[:, 0]/g, label='Accel X')
axs[0].plot(timestamps, acc[:, 1]/g, label='Accel Y')
axs[0].plot(timestamps, acc[:, 2]/g, label='Accel Z')
axs[0].set_ylabel('Acceleration (g)')
axs[0].set_title('Simulated Accelerometer Readings (Gravity + Movement)')
axs[0].legend()
axs[0].grid(True)
# Gyroscope plot
axs[1].plot(timestamps, gyr[:, 0], label='Gyro X')
axs[1].plot(timestamps, gyr[:, 1], label='Gyro Y')
axs[1].plot(timestamps, gyr[:, 2], label='Gyro Z')
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