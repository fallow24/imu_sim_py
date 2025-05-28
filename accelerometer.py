import numpy as np

def movement_readings(poses):
    """
    Calculate accelerometer readings due to the IMU's own 3D movement (linear acceleration),
    ignoring gravity. Each pose: [timestamp, x, y, z, yaw, pitch, roll] (angles in degrees).
    Returns: numpy array of shape (N, 4) with [timestamp, ax, ay, az] in the body frame.
    """
    N = poses.shape[0]
    if N < 2:
        raise ValueError("At least two poses are required to compute IMU readings.")
    
    acc_readings = np.zeros((N, 3))
    
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

    # The first and last two arent meaningful due to central differences,
    # so we set them to the nearest valid reading for continuity.
    if N > 2:
        acc_readings[0] = acc_readings[2]
        acc_readings[1] = acc_readings[2]
        acc_readings[-2] = acc_readings[-3]
        acc_readings[-1] = acc_readings[-2]

    # Add timestamps as first column
    return np.column_stack((poses[:, 0], acc_readings))

def gravity_readings(poses):
    """
    Calculate accelerometer readings due to gravity for a sequence of 6-DoF poses.
    Each pose: [timestamp, x, y, z, yaw, pitch, roll] (angles in degrees).
    Returns: numpy array of shape (N, 4) with [timestamp, gx, gy, gz] in the body frame.
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

    acc_readings = np.array(acc_readings)
    return np.column_stack((poses[:, 0], acc_readings))

def readings(poses, 
             bias=np.array([0, 0, 0]), 
             random_walk=0.00002, 
             noise_density=0.00098,
             saturation=8*9.80665,  # 8g saturation
             random_seed=None):
    """
    Returns simulated accelerometer readings (gravity + movement) with bias and noise.
    Noise is modelled as white Gaussian, bias is modelled as random walk, according to kalibr:
    https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
    Parameters:
        poses: Nx7 pose array [timestamp, x, y, z, yaw, pitch, roll]
        bias: 1x3 array constant bias (m/s^2)
        random_walk: Accelerometer random walk (m/s^3/sqrt(Hz))
        noise_density: standard deviation of measurement noise (m/s^2/sqrt(Hz))
        saturation: maximum absolute value of accelerometer readings (m/s^2)
        random_seed: for reproducibility
    Returns: numpy array of shape (N, 4) with [timestamp, ax, ay, az] in m/s^2 in the body frame.
    """
    N = poses.shape[0]
    if N < 2:
        raise ValueError("At least two poses are required to compute IMU readings.") 

    if random_seed is not None:
        np.random.seed(random_seed)

    acc = gravity_readings(poses)[:, 1:] + movement_readings(poses)[:, 1:]
    N, D = acc.shape

    # Uses pose timestamps to calculate dts
    dt = np.diff(poses[:, 0], prepend=(poses[0,0] - poses[1,0]))

    # Measurement noise (white Gaussian)
    noise = (noise_density / np.sqrt(dt))[:, None] * np.random.normal(0, 1, size=(N, D)) 

    # Bias random walk (brownian motion) 
    drift = random_walk * np.sqrt(dt)[:, None] * np.cumsum(np.random.normal(0, 1, size=(N, D)), axis=0)

    acc_total = np.clip(acc + bias + drift + noise, -saturation, saturation)
    
    # Add timestamps as first column
    return np.column_stack((poses[:, 0], acc_total))