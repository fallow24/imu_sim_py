import numpy as np
import matplotlib.pyplot as plt

# internal
import trajectories as trj
import accelerometer as accel
import gyroscope as gyro
from bag2pose import load_rosbag_poses # if you need ROS1 bag support
from utils import *

# Setup simulation
# Example poses: [timestamp, x, y, z, yaw, pitch, roll]
# Generate a 3D trochoid trajectory for a ball rolling without slipping
duration = 10.0  # seconds
fs = 100         # Hz (sampling time)
N = int(duration * fs)                  
poses = trj.generate_trochoid_forward(duration, fs, 0.29, 0.14, 6)
# poses = trj.generate_rotating_disc(10, 100, 0.5, 12)
# poses = trj.generate_lift_motion(10, 100, 10)

## Example load real world trajectory 
# poses = trj.load_rosbag_poses("/home/fabi/Documents/Bagfiles/Test_accel_2025-05-23.bag", "/lkf/pose", tend=45)
## This usually needs filtering before calculating IMU readings, which is based on derivatives.
# poses = savgol_filter(poses, window_length=11, polyorder=2, axis=0)

acc = accel.readings(poses) # = accel.movement_readings(poses) + accel.gravity_readings(poses)
gyr = gyro.readings(poses)

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

# Draw the trajectory as a red line
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(poses[:, 1], poses[:, 2], poses[:, 3], 'r-', label='Trajectory')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory with Pose Coordinate Systems')
ax.legend()

# Show all the figures
plt.tight_layout()
plt.show()