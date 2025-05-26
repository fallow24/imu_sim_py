import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import medfilt 

def remove_outliers_zscore(data, threshold=1):
    data = data.copy()
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z = np.abs((data - mean) / std)
    # Replace outliers in each column with the column mean
    for i in range(data.shape[1]):
        outliers = z[:, i] > threshold
        data[outliers, i] = mean[i]
    return data

def _quaternion_to_eulerXYZ(qx, qy, qz, qw):
    """
    Convert quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw) in radians 
    using XYZ convention which reflects ROS.
    """
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.pi/2 * np.sign(sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw