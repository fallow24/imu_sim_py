import numpy as np
from scipy.spatial.transform import Rotation as R

def readings(poses, 
             bias=np.array([0,0,0]), 
             random_walk=0.0002, 
             noise_density=0.005, 
             saturation=2000*3.14159/180, # 2000 deg/s saturation
             random_seed=None,
             scale=1.0):
    """
    Returns simulated gyroscope readings (angular velocity in body frame) with bias and noise.
    Noise is modelled as white Gaussian, bias is modelled as random walk, according to kalibr:
    https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
    Parameters:
        poses: Nx7 pose array [timestamp, x, y, z, yaw, pitch, roll]
        bias: 1x3 array constant bias (rad/s)
        random_walk: Gyroscope random walk (rad/s^2/sqrt(Hz))
        noise_density: standard deviation of measurement noise (rad/s/sqrt(Hz))
        saturation: saturation limit for the gyroscope readings (rad/s)
        random_seed: for reproducibility
    Returns: numpy array of shape (N, 4) with [timestamp, wx, wy, wz] in rad/s in the body frame.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    N = poses.shape[0]
    if N < 2:
        raise ValueError("At least two poses are required to compute IMU readings.")

    # Compute orientation quaternions for each pose
    angles = np.deg2rad(poses[:, 4:7])  # yaw, pitch, roll in radians
    quats = R.from_euler('zyx', np.rad2deg(angles), degrees=True)

    # Compute angular velocity in body frame using quaternion difference
    gyro = np.zeros((N, 3))
    for i in range(1, N):
        dt = poses[i, 0] - poses[i-1, 0]
        if dt <= 0:
            continue
        dq = quats[i-1].inv() * quats[i]
        omega = dq.as_rotvec() / dt
        gyro[i] = omega
    gyro[0] = gyro[1]  # first value copy for continuity

    # The first and last two aren't meaningful due to central differences,
    # so we set them to the nearest valid reading for continuity.
    # if N > 2:
    #     gyro[-2] = gyro[-3]
    #     gyro[-1] = gyro[-2]

    # Uses pose timestamps to calculate dts
    dt = np.diff(poses[:, 0], prepend=(poses[0,0] - poses[1,0]))

    # Measurement noise (white Gaussian)
    N, D = gyro.shape
    noise = (noise_density / np.sqrt(dt))[:, None] * np.random.normal(0, 1, size=(N, D)) 

    # Bias random walk (brownian motion) 
    drift = random_walk * np.sqrt(dt)[:, None] * np.cumsum(np.random.normal(0, 1, size=(N, D)), axis=0)

    gyro_total = np.clip(scale*gyro + bias + drift + noise, -saturation, saturation)

    # Add timestamps as first column
    return np.column_stack((poses[:, 0], gyro_total))