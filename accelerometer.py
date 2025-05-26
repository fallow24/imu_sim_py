import numpy as np

def movement_readings(poses):
    """
    Calculate accelerometer readings due to the IMU's own 3D movement (linear acceleration),
    ignoring gravity. Each pose: [timestamp, x, y, z, yaw, pitch, roll] (angles in degrees).
    Returns: numpy array of shape (N, 3) with linear acceleration in the body frame.
    """
    N = poses.shape[0]
    acc_readings = np.zeros((N, 3))
    dt = np.diff(poses[:, 0])
    
    # Compute velocities in world frame
    vel = np.zeros((N, 3))
    dt_c = poses[2:, 0] - poses[:-2, 0]  # central time differences
    vel[1:-1] = (poses[2:, 1:4] - poses[:-2, 1:4]) / dt_c[:, None]
    vel[0] = vel[1]
    vel[-1] = vel[-2]

    # Compute accelerations in world frame
    acc_world = np.zeros((N, 3))
    acc_world[1:-1] = (vel[2:] - vel[:-2]) / dt_c[:, None]
    acc_world[0] = acc_world[1]
    acc_world[-1] = acc_world[-2]

    for i, pose in enumerate(poses):
        _, _, _, _, yaw, pitch, roll = pose
        # Convert angles to radians
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

        # Rotation matrices
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [ 0,             1, 0           ],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0,           0          ],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        # NED to body: R = Rx * Ry * Rz
        R = Rx @ Ry @ Rz
        # Transform acceleration from world to body frame
        acc_body = R.T @ acc_world[i]
        acc_readings[i] = acc_body

    # Assume to have initial acceleration for continuity
    acc_readings[0] = acc_readings[2]
    acc_readings[1] = acc_readings[2]
    return acc_readings

def gravity_readings(poses):
    """
    Calculate accelerometer readings due to gravity for a sequence of 6-DoF poses.
    Each pose: [timestamp, x, y, z, yaw, pitch, roll] (angles in degrees).
    Returns: numpy array of shape (N, 3) with gravity in the body frame.
    """
    g = 9.80665  # gravity magnitude in m/s^2
    gravity_ned = np.array([0, 0, g])  # gravity in NED frame (down is positive)
    acc_readings = []

    for pose in poses:
        _, _, _, _, yaw, pitch, roll = pose
        # Convert angles to radians
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

        # Rotation matrices
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [ 0,             1, 0           ],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0,           0          ],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        # NED to body: R = Rx * Ry * Rz
        R = Rx @ Ry @ Rz
        # Gravity in body frame
        gravity_body = R.T @ gravity_ned  # transpose for NED->body
        acc_readings.append(gravity_body)

    return np.array(acc_readings)

def readings(poses):
    return gravity_readings(poses) + movement_readings(poses)