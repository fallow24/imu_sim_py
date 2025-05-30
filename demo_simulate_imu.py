import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# internalw
import trajectories as trj
import accelerometer as accel
import gyroscope as gyro
import filter as filt 
from utils import *
# (optional) if you require ROS1 bag support
from bag2pose import load_rosbag_poses

## Setup simulation

# Example poses: [timestamp, x, y, z, yaw, pitch, roll]
duration = 10.0  # seconds
fs = 125    # Hz (sampling time)     

poses = trj.generate_rotating_disc(duration, fs, 0.13, 6)  # Generate a rotating disc trajectory
#poses = trj.generate_lift_motion(duration, fs, 10)
#poses = trj.generate_trochoid_forward(duration, fs, 0.145, 0.13, 6)
#poses = load_rosbag_poses("/home/fabi/Documents/Bagfiles/Test_accel_2025-05-23.bag", "/lkf/pose", tend=45)

#acc = accel.gravity_readings(poses)  # shape (N, 4): [timestamp, ax, ay, az]
acc = accel.readings(poses, random_seed=43)#, random_walk=0, noise_density=0)  # shape (N, 4): [timestamp, ax, ay, az]
gyr = gyro.readings(poses, random_seed=43)#, random_walk=0, noise_density=0)   # shape (N, 4): [timestamp, gx, gy, gz]

# Apply complementary filter to estimate orientation
quats = filt.imujasper(acc, gyr, gain=0.02, alpha=0.8, autogain=0.01, gain_min=0.01)
#quats = filt.complementary(acc, gyr, alpha=0.8)  # shape (N, 5): [timestamp, x, y, z, w]
#quats = filt.imujasper(acc, gyr, gain=0, alpha=0.8, autogain=0, gain_min=0)  # shape (N, 5): [timestamp, x, y, z, w]
## FIGURE 1

# Plot the accelerometer data using timestamp as x axis
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
timestamps = acc[:, 0]  # Use timestamps from acc (or gyr, both are same as poses)

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
axs[1].set_ylabel('Angular Rate (rad/s)')
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

## FIGURE 3
# Convert estimated quaternions to euler angles (yaw, pitch, roll)
# quats: (N, 5) [timestamp, x, y, z, w]
est_rot = R.from_quat(quats[:, 1:5])
est_euler = est_rot.as_euler('zyx', degrees=True)  # yaw, pitch, roll

# Ground truth euler angles from poses (in degrees)
gt_euler_orig = poses[:, 4:7]  # yaw, pitch, roll

# Convert ground truth to rotation objects and then to est_euler's convention
gt_rot = R.from_euler('zyx', gt_euler_orig, degrees=True)
gt_euler = gt_rot.as_euler('zyx', degrees=True)  # Now in the same convention as est_euler

# Unwrap yaw and roll angles to avoid discontinuities and center to [-180, 180]
est_yaw = (est_euler[:, 0] + 180) % 360 - 180
gt_yaw = (gt_euler[:, 0] + 180) % 360 - 180

# Unwrap roll angles and center to [-180, 180]
est_roll = (est_euler[:, 2] + 180) % 360 - 180
gt_roll = (gt_euler[:, 2] + 180) % 360 - 180

# Flip ground truth roll to match estimated roll convention
gt_roll = -gt_roll

# Optionally, wrap pitch to [-180, 180] for both
est_pitch = (est_euler[:, 1] + 180) % 360 - 180
gt_pitch = (gt_euler[:, 1] + 180) % 360 - 180

fig3, axs3 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot yaw, pitch, roll (estimated and ground truth)
axs3[0].plot(poses[:, 0], est_yaw, label='Yaw (est)')
axs3[0].plot(poses[:, 0], est_pitch, label='Pitch (est)')
axs3[0].plot(poses[:, 0], est_roll, label='Roll (est)')
axs3[0].plot(poses[:, 0], gt_yaw, 'k--', label='Yaw (GT)')
axs3[0].plot(poses[:, 0], gt_pitch, 'r--', label='Pitch (GT)')
axs3[0].plot(poses[:, 0], gt_roll, 'b--', label='Roll (GT)')
axs3[0].set_ylabel('Angle (deg)')
axs3[0].set_title('Estimated vs Ground Truth Yaw, Pitch, Roll')
axs3[0].legend()
axs3[0].grid(True)

# Compute rotation error (angle between estimated and ground truth orientation)
gt_rot = R.from_euler('zyx', gt_euler, degrees=True)
rot_diff = gt_rot.inv() * est_rot
rot_err_deg = np.abs(rot_diff.magnitude()) * 180 / np.pi  # rotation angle in degrees

axs3[1].plot(poses[:, 0], rot_err_deg, label='Rotation Error')
axs3[1].set_xlabel('Time (s)')
axs3[1].set_ylabel('Error (deg)')
axs3[1].set_title('Rotation Error (Angle between estimated and ground truth)')
axs3[1].legend()
axs3[1].grid(True)

# Show all the figures
plt.tight_layout()
plt.show()