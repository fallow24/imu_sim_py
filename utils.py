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

def _quat_multiply(q, r):
    """
    Quaternion multiplication (Hamilton product). Both q and r are [x, y, z, w].
    """
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = r
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])