import numpy as np
import matplotlib.pyplot as plt

# internalw
import trajectories as trj
import accelerometer as accel
import gyroscope as gyro
from utils import *
# (optional) if you require ROS1 bag support
from bag2pose import load_rosbag_poses

## Setup simulation

# Example poses: [timestamp, x, y, z, yaw, pitch, roll]
duration = 10.0  # seconds
fs = 200     # Hz (sampling time)     
poses = trj.generate_stationary_pose(duration=10, fs=200);          
poses = trj.generate_trochoid_forward(duration, fs, 0.145, 0.13, 6)
#poses = load_rosbag_poses("/home/fabi/Documents/Bagfiles/Test_accel_2025-05-23.bag", "/lkf/pose", tend=45)

acc = accel.readings(poses)
gyr = gyro.readings(poses)
gyr = np.rad2deg(gyr)  # convert to deg/s

## FIGURE 1

# Plot the accelerometer data using timestamp as x axis
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
timestamps = poses[:, 0]

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

## FIGURE 2

## Draw the trajectory as a red line
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(poses[:, 1], poses[:, 2], poses[:, 3], 'b--', label='Trajectory')
# Plot starting pose as a green dot
ax.scatter(poses[0, 1], poses[0, 2], poses[0, 3], c='g', s=100, label='Start Position')
# Plot the end pose as a blue dot
ax.scatter(poses[-1, 1], poses[-1, 2], poses[-1, 3], c='r', s=100, label='End Position')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory')
ax.legend()

# Show all the figures
plt.tight_layout()
plt.show()