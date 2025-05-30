import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from utils import _quat_multiply

def complementary(acc, gyr, alpha=0.98):
    """
    Simple complementary filter for IMU orientation estimation.
    Args:
        acc: (N, 4) accelerometer readings [timestamp, ax, ay, az]
        gyr: (N, 4) gyroscope readings [timestamp, gx, gy, gz]
        alpha: filter coefficient (0 < alpha < 1), higher = more gyro trust
    Returns:
        quats: (N, 5) array of [timestamp, x, y, z, w]
    """
    N = acc.shape[0]
    quats = np.zeros((N, 5))
    dt = np.diff(acc[:, 0], prepend=acc[0, 0])

    # Initialize orientation from accelerometer (assume flat start)
    acc0 = acc[0, 1:4] / np.linalg.norm(acc[0, 1:4])
    pitch0 = np.arcsin(-acc0[0])
    roll0 = np.arctan2(acc0[1], acc0[2])
    r = R.from_euler('zyx', [0, pitch0, roll0])
    quats[0, 0] = acc[0, 0]
    quats[0, 1:] = r.as_quat()  # [x, y, z, w]

    for i in range(1, N):
        # Integrate gyro to propagate orientation
        omega = gyr[i, 1:4] * dt[i]
        r_gyro = R.from_rotvec(omega)
        r_prev = R.from_quat(quats[i-1, 1:])
        r_pred = r_prev * r_gyro

        # Compute pitch/roll from accelerometer, keep yaw from prediction
        acc_i = acc[i, 1:4] / np.linalg.norm(acc[i, 1:4])
        pitch_acc = np.arcsin(-acc_i[0])
        roll_acc = np.arctan2(acc_i[1], acc_i[2])
        yaw_pred, _, _ = r_pred.as_euler('zyx', degrees=False)
        r_acc = R.from_euler('zyx', [yaw_pred, pitch_acc, roll_acc])

        # Complementary filter: combine gyro and accel (slerp)
        slerp = Slerp([0, 1], R.concatenate([r_pred, r_acc]))
        r_filt = slerp([1 - alpha])[0]
        quats[i, 0] = acc[i, 0]
        quats[i, 1:] = r_filt.as_quat()

    return quats

def madgwick(acc, gyr, beta=0.1):
    """
    Madgwick filter for IMU orientation estimation (no magnetometer).
    Args:
        acc: (N, 4) accelerometer readings [timestamp, ax, ay, az]
        gyr: (N, 4) gyroscope readings [timestamp, gx, gy, gz]
        beta: filter gain (typical: 0.01 - 0.2)
    Returns:
        quats: (N, 5) array of [timestamp, x, y, z, w]
    """
    N = acc.shape[0]
    quats = np.zeros((N, 5))
    # Initialize quaternion as [x, y, z, w]
    q = np.array([0, 0, 0, 1], dtype=np.float64)
    quats[0, 0] = acc[0, 0]
    quats[0, 1:] = q

    for i in range(1, N):
        dt = acc[i, 0] - acc[i-1, 0]
        if dt <= 0:
            quats[i, 0] = acc[i, 0]
            quats[i, 1:] = q
            continue

        # Normalize accelerometer
        a = acc[i, 1:4]
        if np.linalg.norm(a) == 0:
            quats[i, 0] = acc[i, 0]
            quats[i, 1:] = q
            continue
        a = a / np.linalg.norm(a)

        # Gyroscope in rad/s
        gx, gy, gz = gyr[i, 1:4]

        # Quaternion derivative from gyroscope (Hamilton product)
        q_dot = 0.5 * _quat_multiply(q, np.array([gx, gy, gz, 0]))

        # Gradient descent correction (accelerometer only)
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - a[0],
            2*(q[0]*q[1] + q[2]*q[3]) - a[1],
            2*(0.5 - q[1]**2 - q[2]**2) - a[2]
        ])
        J = np.array([
            [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
            [ 2*q[1],  2*q[0],  2*q[3], 2*q[2]],
            [     0, -4*q[1],  -4*q[2],     0]
        ])
        step = J.T @ f
        norm_step = np.linalg.norm(step)
        if norm_step > 1e-8:
            step = step / norm_step
        else:
            step = np.zeros_like(step)

        # Apply correction
        q_dot -= beta * step

        # Integrate to yield new quaternion
        q += q_dot * dt
        q /= np.linalg.norm(q)

        quats[i, 0] = acc[i, 0]
        quats[i, 1:] = q

    return quats

def imujasper(acc, gyr, gain=0.2, alpha=0.02, autogain=0.0, gain_min=0.02):
    """
    Hybrid Madgwick + Complementary filter as described in the provided C++ code.
    Args:
        acc: (N, 4) accelerometer readings [timestamp, ax, ay, az]
        gyr: (N, 4) gyroscope readings [timestamp, gx, gy, gz]
        gain: base gain for gradient descent step (Madgwick)
        alpha: complementary filter blending factor (0 < alpha < 1)
        autogain: if >0, adapt gain based on rotation rate
        gain_min: minimum gain if autogain is used
    Returns:
        quats: (N, 5) array of [timestamp, x, y, z, w]
    """
    N = acc.shape[0]
    quats = np.zeros((N, 5))
    # Initialize quaternion as [x, y, z, w]
    q = np.array([1, 0, 0, 0], dtype=np.float64)  # [w, x, y, z]
    quats[0, 0] = acc[0, 0]
    quats[0, 1:] = np.array([q[1], q[2], q[3], q[0]])  # [x, y, z, w]

    for i in range(1, N):
        dt = acc[i, 0] - acc[i-1, 0]
        if dt <= 0:
            quats[i, 0] = acc[i, 0]
            quats[i, 1:] = np.array([q[1], q[2], q[3], q[0]])
            continue

        q0, q1, q2, q3 = q  # [w, x, y, z]
        gx, gy, gz = gyr[i, 1:4]
        ax, ay, az = acc[i, 1:4]

        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        # Adaptive gain
        factorX = min((0.5 * gx) ** 2, 1)
        factorY = min((0.5 * gy) ** 2, 1)
        factorZ = min((0.5 * gz) ** 2, 1)
        gain_ = gain
        alpha_ = alpha
        if autogain > 0:
            maxFac = max(factorX, factorY, factorZ)
            gain_ = max(autogain * maxFac, gain_min)
            if maxFac < 0.1:
                maxFac = 0
            alpha_ = autogain * 0.1 * (1 - maxFac)

        # Accelerometer normalization
        if ax == 0.0 and ay == 0.0 and az == 0.0:
            axNom, ayNom, azNom = 0.0, 0.0, 0.0
        else:
            recipNorm = 1.0 / np.sqrt(ax * ax + ay * ay + az * az)
            axNom = ax * recipNorm
            ayNom = ay * recipNorm
            azNom = az * recipNorm

        # Auxiliary variables
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        # Gradient descent step
        s0 = _4q0 * q2q2 + _2q2 * axNom + _4q0 * q1q1 - _2q1 * ayNom
        s1 = _4q1 * q3q3 - _2q3 * axNom + 4.0 * q0q0 * q1 - _2q0 * ayNom - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * azNom
        s2 = 4.0 * q0q0 * q2 + _2q0 * axNom + _4q2 * q3q3 - _2q3 * ayNom - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * azNom
        s3 = 4.0 * q1q1 * q3 - _2q1 * axNom + 4.0 * q2q2 * q3 - _2q2 * ayNom

        norm_s = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if norm_s > 1e-8:
            s0 /= norm_s
            s1 /= norm_s
            s2 /= norm_s
            s3 /= norm_s
        else:
            s0 = s1 = s2 = s3 = 0.0

        # Apply feedback step
        qDot1 -= gain_ * s0
        qDot2 -= gain_ * s1
        qDot3 -= gain_ * s2
        qDot4 -= gain_ * s3

        # Integrate rate of change of quaternion
        q0 += qDot1 * dt
        q1 += qDot2 * dt
        q2 += qDot3 * dt
        q3 += qDot4 * dt

        # Normalize quaternion
        norm = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if norm > 1e-8:
            q0 /= norm
            q1 /= norm
            q2 /= norm
            q3 /= norm
        else:
            q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0

        # Complementary filter step
        acc_roll = np.arctan2(ayNom, azNom)
        acc_pitch = np.arctan2(-axNom, np.sqrt(ayNom * ayNom + azNom * azNom))

        # Get RPY from quaternion
        try:
            r = R.from_quat([q1, q2, q3, q0])
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        except ValueError:
            roll, pitch, yaw = 0.0, 0.0, 0.0

        # Gimbal lock avoidance
        if np.abs(q2 * q1 + q0 * q3) - 0.5 < 0.01:
            acc_roll = roll

        # Quaternion from acc_roll, acc_pitch, yaw
        r_acc = R.from_euler('xyz', [acc_roll, acc_pitch, yaw])
        q_acc = r_acc.as_quat()  # [x, y, z, w]

        # Slerp between current and acc quaternion
        r_cur = R.from_quat([q1, q2, q3, q0])
        slerp = Slerp([0, 1], R.concatenate([r_cur, r_acc]))
        r_next = slerp([alpha_])[0]
        q_next = r_next.as_quat()  # [x, y, z, w]

        # Update q for next iteration (convert to [w, x, y, z])
        q = np.array([q_next[3], q_next[0], q_next[1], q_next[2]])

        # Normalize again for safety
        q /= np.linalg.norm(q)

        quats[i, 0] = acc[i, 0]
        quats[i, 1:] = np.array([q[1], q[2], q[3], q[0]])  # [x, y, z, w]

    return quats