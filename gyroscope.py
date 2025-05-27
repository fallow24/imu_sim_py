import numpy as np

def readings(poses, 
             bias=np.array([0,0,0]), 
             random_walk=0.0002, 
             noise_density=0.005, 
             saturation=2000*3.14159/180, # 2000 deg/s saturation
             random_seed=None):
    """
    Returns simulated accelerometer readings (gravity + movement) with bias and noise.
    Noise is modelled as white Gaussian, bias is modelled as random walk, according to kalibr:
    https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
    Parameters:
        poses: Nx7 pose array [timestamp, x, y, z, yaw, pitch, roll]
        bias: 1x3 array constant bias (rad/s)
        random_walk: Accelerometer random walk (rad/s^2/sqrt(Hz))
        noise_density: standard deviation of measurement noise (rad/s/sqrt(Hz))
        saturation: saturation limit for the gyroscope readings (rad/s)
        random_seed: for reproducibility
    Returns: numpy array of shape (N, 3) with angular velocity [wx, wy, wz] in rad/s in the body frame.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    N = poses.shape[0]
    if N < 2:
        raise ValueError("At least two poses are required to compute IMU readings.")
    
    gyro = np.zeros((N, 3))
    angles = np.unwrap(np.deg2rad(poses[:, 4:7]), axis=0)  # yaw, pitch, roll in radians

    # Compute angular velocity in world frame (finite differences)
    ang_vel_world = np.zeros((N, 3))
    dt_c = poses[2:, 0] - poses[:-2, 0]
    ang_vel_world[1:-1] = (angles[2:] - angles[:-2]) / dt_c[:, None]
    ang_vel_world[0] = ang_vel_world[1]
    ang_vel_world[-1] = ang_vel_world[-2]

    # Convert angular velocity from world to body frame
    for i, pose in enumerate(poses):
        yaw, pitch, roll = angles[i]
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
        R = Rx @ Ry @ Rz
        omega_body = R.T @ ang_vel_world[i]
        gyro[i] = omega_body   
        
    # The first and last two arent meaningful due to central differences,
    # so we set them to the nearest valid reading for continuity.
    if N > 2:
        gyro[0] = gyro[2]
        gyro[1] = gyro[2]
        gyro[-2] = gyro[-3]
        gyro[-1] = gyro[-2]

    # Uses pose timestamps to calculate dts
    dt = np.diff(poses[:, 0], prepend=(poses[0,0] - poses[1,0]))

    # Measurement noise (white Gaussian)
    N, D = gyro.shape
    noise = (noise_density / np.sqrt(dt))[:, None] * np.random.normal(0, 1, size=(N, D)) 
    
    # Bias random walk (brownian motion) 
    drift = random_walk * np.sqrt(dt)[:, None] * np.cumsum(np.random.normal(0, 1, size=(N, D)), axis=0)
    
    return np.clip(gyro + bias + drift + noise, -saturation, saturation)
