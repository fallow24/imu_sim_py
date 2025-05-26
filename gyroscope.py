import numpy as np

def readings(poses):
    """
    Calculate gyroscope readings (angular velocity in body frame) for a sequence of 6-DoF poses.
    Each pose: [timestamp, x, y, z, yaw, pitch, roll] (angles in degrees).
    Returns: numpy array of shape (N, 3) with angular velocity [wx, wy, wz] in deg/s in the body frame.
    """
    N = poses.shape[0]
    gyro = np.zeros((N, 3))
    dt = np.diff(poses[:, 0])
    # Convert angles to radians for computation
    angles = np.unwrap(np.deg2rad(poses[:, 4:7]), axis=0)  # yaw, pitch, roll in radians

    # Compute angular velocity in world frame (finite differences)
    ang_vel_world = np.zeros((N, 3))
    dt_c = poses[2:, 0] - poses[:-2, 0]
    ang_vel_world[1:-1] = (angles[2:] - angles[:-2]) / dt_c[:, None]
    ang_vel_world[0] = ang_vel_world[1]  # assume continuity
    ang_vel_world[-1] = ang_vel_world[-2]  # assume continuity

    # Convert angular velocity from world to body frame
    for i, pose in enumerate(poses):
        yaw, pitch, roll = angles[i]
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
        # Transform angular velocity from world to body frame
        omega_body = R.T @ ang_vel_world[i]
        # Convert back to deg/s for output
        gyro[i] = np.rad2deg(omega_body)

    # Handle initial values for continuity
    gyro[0] = gyro[2]
    gyro[1] = gyro[2]
    return gyro