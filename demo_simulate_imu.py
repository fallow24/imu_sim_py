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

duration = 15  # seconds
fs = 125    # Hz (sampling time) 
R_sphere = 0.145
t_imu = np.array([0.13, 0.13, 0.02])

# SOME EXAMPLE TRAJECTORIES
# Example poses: [timestamp, x, y, z, yaw, pitch, roll]
#poses = trj.generate_lift_motion(duration, fs, 10)
#poses = trj.generate_rotating_disc(duration, fs, 0.13, 6)  # Generate a rotating disc trajectory
#poses = trj.generate_trochoid_curved(duration, fs, 0.145, 0.13, 6, turn_period=2, turn_angle_deg=100)
#poses = load_rosbag_poses("/home/fabi/Documents/Bagfiles/Test_accel_2025-05-23.bag", "/lkf/pose", tend=45)

# Create a sin wave as np array with duration and sampling frequency fs
omega_input = np.zeros((fs * duration, 4))  # [timestamp, wx, wy, wz]
omega_input[:, 0] = np.linspace(0, duration, fs * duration)  # timestamps
omega_input[:, 1] = np.sin(np.linspace(0, 2 * np.pi * 1.5, fs * duration)) * 0.5 + 0.5 # wx
omega_input[:, 2] = np.sin(np.linspace(0, 2 * np.pi * 1, fs * duration)) * 2 + 2  # wy
omega_input[:, 3] = np.sin(np.linspace(0, 2 * np.pi * 0.8, fs * duration)) * 0.7 - 1 # wz

#poses = trj.generate_trochoid_forward(duration, fs, 0.145, 0.13, 6)  # Generate a trochoid trajectory
poses = trj.generate_trochoid(omega_vec=omega_input, R_ball=R_sphere, offset_vec=t_imu)
initial_quat = R.from_euler('zyx', poses[0, 4:7], degrees=True).as_quat()  # [x, y, z, w]

#acc = accel.gravity_readings(poses)
acc = accel.readings(poses)
gyr = gyro.readings(poses, bias=np.array([0.01, -0.02, 0.015]))

# COMPARISON OF WHAT RUNS ON THE PROTOTYPE (BAD ACTUALLY)
#quats = filt.imujasper(acc, gyr, gain=0.2, alpha=0.02, autogain=0.2, gain_min=0.1, initial_quat=initial_quat) 
#quats2 = filt.imujasper(acc, gyr, gain=0.2, alpha=0.02, autogain=0.2, gain_min=0.1, t_imu=t_imu, initial_quat=initial_quat) 

# COMPARISON OF SIMPLE COMPLEMENTARY FILTER
#quats = filt.imujasper(acc, gyr, gain=0.0, alpha=0.05, autogain=0, gain_min=0, initial_quat=initial_quat) 
quats = filt.imujasper(acc, gyr, gain=0.0, alpha=0.05, autogain=0, gain_min=0, initial_quat=initial_quat) 
quats2 = filt.imujasper(acc, gyr, gain=0.0, alpha=0.05, autogain=0, gain_min=0, t_imu=t_imu, initial_quat=initial_quat) 
# quats3 = filt.imujasper(acc, gyr, gain=0.0, alpha=0.05, autogain=0, gain_min=0, initial_quat=initial_quat) 
# quats4 = filt.imujasper(acc, gyr, gain=0.0, alpha=0.1, autogain=0, gain_min=0, initial_quat=initial_quat) 
# quats5 = filt.imujasper(acc, gyr, gain=0.0, alpha=0.8, autogain=0, gain_min=0, initial_quat=initial_quat) 

## FIGURE 1

# Plot the accelerometer data using timestamp as x axis
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
timestamps = acc[:, 0]  # Use timestamps from acc (or gyr, both are same as poses)


# Accelerometer plot
g = 9.80665
acc = accel.movement_readings(poses)
axs[0].plot(timestamps, acc[:, 1]/g, label='Accel X')
axs[0].plot(timestamps, acc[:, 2]/g, label='Accel Y')
axs[0].plot(timestamps, acc[:, 3]/g, label='Accel Z')
axs[0].set_ylabel('Acceleration (g)')
axs[0].set_title('Simulated Accelerometer Readings (ONLY Movement)')
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
# Ensure quaternion sign consistency for error calculation and plotting
est_quat = quats[:, 1:5]
gt_quat = R.from_euler('zyx', poses[:, 4:7], degrees=True).as_quat()
for i in range(len(est_quat)):
    if np.dot(est_quat[i], gt_quat[i]) < 0:
        est_quat[i] = -est_quat[i]

est_rot = R.from_quat(est_quat)
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

# fig3, axs3 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# # Plot yaw, pitch, roll (estimated and ground truth)
# axs3[0].plot(poses[:, 0], est_yaw, 'r', label='Yaw (est)')
# axs3[0].plot(poses[:, 0], est_pitch, 'g', label='Pitch (est)')
# axs3[0].plot(poses[:, 0], est_roll, 'b', label='Roll (est)')
# axs3[0].plot(poses[:, 0], gt_yaw, 'r--', label='Yaw (GT)')
# axs3[0].plot(poses[:, 0], gt_pitch, 'g--', label='Pitch (GT)')
# axs3[0].plot(poses[:, 0], gt_roll, 'b--', label='Roll (GT)')
# axs3[0].set_ylabel('Angle (deg)')
# axs3[0].set_title('Estimated vs Ground Truth Yaw, Pitch, Roll')
# axs3[0].legend()
# axs3[0].grid(True)

# # Compute rotation error (angle between estimated and ground truth orientation)
# # Construct delta rotation, convert to angle-axis, and use the angle as the error
# rot_err_deg = np.zeros(len(est_quat))
# for i in range(len(est_quat)):
#     # Align quaternion signs
#     if np.dot(est_quat[i], gt_quat[i]) < 0:
#         q_est = -est_quat[i]
#     else:
#         q_est = est_quat[i]
#     q_gt = gt_quat[i]
#     # Build delta rotation
#     r_est = R.from_quat(q_est)
#     r_gt = R.from_quat(q_gt)
#     delta_rot = r_gt.inv() * r_est
#     # Angle-axis: use the angle as the error
#     rot_err_deg[i] = np.abs(delta_rot.magnitude()) * 180 / np.pi

# # Set very small errors to zero for plotting clarity
# rot_err_deg[rot_err_deg < 1e-8] = 0.0

# # Compute rotation error for quats2 (angle between estimated and ground truth orientation)
# est_quat2 = quats2[:, 1:5]
# # Ensure quaternion sign consistency for error calculation and plotting
# for i in range(len(est_quat2)):
#     if np.dot(est_quat2[i], gt_quat[i]) < 0:
#         est_quat2[i] = -est_quat2[i]

# rot_err_deg2 = np.zeros(len(est_quat2))
# for i in range(len(est_quat2)):
#     # Align quaternion signs
#     if np.dot(est_quat2[i], gt_quat[i]) < 0:
#         q_est2 = -est_quat2[i]
#     else:
#         q_est2 = est_quat2[i]
#     q_gt = gt_quat[i]
#     # Build delta rotation
#     r_est2 = R.from_quat(q_est2)
#     r_gt = R.from_quat(q_gt)
#     delta_rot2 = r_gt.inv() * r_est2
#     # Angle-axis: use the angle as the error
#     rot_err_deg2[i] = np.abs(delta_rot2.magnitude()) * 180 / np.pi

# # Set very small errors to zero for plotting clarity
# rot_err_deg2[rot_err_deg2 < 1e-8] = 0.0

# List of all filter results and their labels
quats_list = [
    (quats, 'alpha=0.05 (raw)'),
    (quats2, 'alpha=0.05 (compensated)'),
    # (quats3, 'alpha=0.05'),
    # (quats4, 'alpha=0.1'),
    # (quats5, 'alpha=0.8'),
]

# Prepare colors for plotting
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Plot rotation errors for all filter results in a single plot
plt.figure(figsize=(12, 5))
for idx, (q, label) in enumerate(quats_list):
    est_quat = q[:, 1:5]
    for i in range(len(est_quat)):
        if np.dot(est_quat[i], gt_quat[i]) < 0:
            est_quat[i] = -est_quat[i]
    rot_err_deg = np.zeros(len(est_quat))
    for i in range(len(est_quat)):
        q_est = est_quat[i]
        q_gt = gt_quat[i]
        r_est = R.from_quat(q_est)
        r_gt = R.from_quat(q_gt)
        delta_rot = r_gt.inv() * r_est
        rot_err_deg[i] = np.abs(delta_rot.magnitude()) * 180 / np.pi
    rot_err_deg[rot_err_deg < 1e-8] = 0.0
    plt.plot(poses[:, 0], rot_err_deg, color=colors[idx], label=f'Error ({label})')

plt.xlabel('Time (s)')
plt.ylabel('Error (deg)')
plt.title('Rotation Error (Angle between estimated and ground truth)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()