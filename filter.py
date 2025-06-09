import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from utils import _quat_multiply, skew

def complementary(acc, gyr, alpha=0.98, initial_quat=None):
    """
    Complementary filter as described in the provided C++ code.
    Args:
        acc: (N, 4) accelerometer readings [timestamp, ax, ay, az]
        gyr: (N, 4) gyroscope readings [timestamp, gx, gy, gz]
        alpha: complementary filter blending factor (0 < alpha < 1)
        initial_quat: optional, initial orientation as [x, y, z, w]
    Returns:
        quats: (N, 5) array of [timestamp, x, y, z, w]
    """
    N = acc.shape[0]
    quats = np.zeros((N, 5))
    # --- Initialization ---
    if initial_quat is not None:
        # Use provided initial quaternion (should be [x, y, z, w])
        q = np.array([initial_quat[3], initial_quat[0], initial_quat[1], initial_quat[2]], dtype=np.float64)  # [w, x, y, z]
    else:
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

        # Accelerometer normalization
        if ax == 0.0 and ay == 0.0 and az == 0.0:
            axNom, ayNom, azNom = 0.0, 0.0, 0.0
        else:
            recipNorm = 1.0 / np.sqrt(ax * ax + ay * ay + az * az)
            axNom = ax * recipNorm
            ayNom = ay * recipNorm
            azNom = az * recipNorm

        # Integrate rate of change of quaternion using quaternion multiplication
        omega = np.array([gx, gy, gz])
        theta = np.linalg.norm(omega * dt)
        if theta > 1e-8:
            axis = omega / np.linalg.norm(omega)
            dq = R.from_rotvec(axis * theta)
        else:
            dq = R.from_quat([0, 0, 0, 1])
        r_prev = R.from_quat([q1, q2, q3, q0])
        r_new = r_prev * dq
        q0, q1, q2, q3 = r_new.as_quat()[3], r_new.as_quat()[0], r_new.as_quat()[1], r_new.as_quat()[2]

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
        
        # Slerp between current and acc quaternion
        r_cur = R.from_quat([q1, q2, q3, q0])
        slerp = Slerp([0, 1], R.concatenate([r_cur, r_acc]))
        r_next = slerp([1 - alpha])[0]
        q_next = r_next.as_quat()  # [x, y, z, w]

        # Update q for next iteration (convert to [w, x, y, z])
        q = np.array([q_next[3], q_next[0], q_next[1], q_next[2]])

        # Normalize again for safety
        q /= np.linalg.norm(q)

        quats[i, 0] = acc[i, 0]
        quats[i, 1:] = np.array([q[1], q[2], q[3], q[0]])  # [x, y, z, w]

    return quats

def imujasper(acc, gyr, gain=0.2, alpha=0.02, autogain=0.0, gain_min=0.02, t_imu=None, initial_quat=None):
    """
    Hybrid Madgwick + Complementary filter as described in Jaspers C++ code.
    Args:
        acc: (N, 4) accelerometer readings [timestamp, ax, ay, az]
        gyr: (N, 4) gyroscope readings [timestamp, gx, gy, gz]
        gain: base gain for gradient descent step (Madgwick)
        alpha: complementary filter blending factor (0 < alpha < 1)
        autogain: if >0, adapt gain based on rotation rate
        gain_min: minimum gain if autogain is used
        t_imu: optional, translation parameters: center of rotation -> IMU position [x, y, z]
        initial_quat: optional, initial orientation as [x, y, z, w]
    Returns:
        quats: (N, 5) array of [timestamp, x, y, z, w]
    """
    N = acc.shape[0]
    quats = np.zeros((N, 5))
    # --- Initialization ---
    if initial_quat is not None:
        # Use provided initial quaternion (should be [x, y, z, w])
        q = np.array([initial_quat[3], initial_quat[0], initial_quat[1], initial_quat[2]], dtype=np.float64)  # [w, x, y, z]
    else:
        q = np.array([1, 0, 0, 0], dtype=np.float64)  # [w, x, y, z]
    quats[0, 0] = acc[0, 0]
    quats[0, 1:] = np.array([q[1], q[2], q[3], q[0]])  # [x, y, z, w]
    normal = np.array([0, 0, 1])

    # Apply lowpass filter to angular acceleration
    class Lowpass:
        def __init__(self):
            self.smoothed = 0.0
            self.alpha = 0.0

        def filter(self, inp):
            self.smoothed = self.smoothed - self.alpha * (self.smoothed - inp)
            return self.smoothed

        def set_freq(self, f_cut, f_sample):
            c = np.cos(2 * np.pi * f_cut / f_sample)
            self.alpha = np.sqrt(c * c - 4 * c + 3) + c - 1

    lowpass = Lowpass()
    lowpass.set_freq(5, 125)
    lastgx = gyr[0, 1]
    lastgy = gyr[0, 2]
    lastgz = gyr[0, 3]
    ur = -t_imu if t_imu is not None else np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    # To be used for: Euler acceleration and translational acceleration (due to ball movement)
    r = np.sqrt(t_imu[0] * t_imu[0] + t_imu[1] * t_imu[1] + t_imu[2] * t_imu[2]) if t_imu is not None else 0.0
    Rsphere = 0.145

    comp_x = []
    comp_y = []
    comp_z = []

    for i in range(1, N):
        dt = acc[i, 0] - acc[i-1, 0]
        if dt <= 0:
            quats[i, 0] = acc[i, 0]
            quats[i, 1:] = np.array([q[1], q[2], q[3], q[0]])
            continue

        q0, q1, q2, q3 = q  # [w, x, y, z]
        gx, gy, gz = gyr[i, 1:4]
        ax, ay, az = acc[i, 1:4]
        invRot = R.from_quat([q1, q2, q3, q0]).inv().as_matrix() # [x y z w]

        # Compensate centripetal and tangential acceleration
        if t_imu is not None:
            gdot_x = (gx - lastgx) / dt
            gdot_y = (gy - lastgy) / dt
            gdot_z = (gz - lastgz) / dt
            gdot = np.array([gdot_x, gdot_y, gdot_z])
            lastgx = gx
            lastgy = gy
            lastgz = gz 
            wdot = lowpass.filter(gdot)

            uomega = np.array([gx, gy, gz]) / np.linalg.norm(np.array([gx, gy, gz])) if np.linalg.norm(np.array([gx, gy, gz])) > 1e-8 else np.array([0.0, 0.0, 0.0])
            utheta = np.cross(uomega, ur)
            g_cross_r = np.cross(gyr[i, 1:4], ur) 
            
            # Use current quaternion to rotate the global (0,0,-1) vector to the local frame
            g_rotated = invRot @ np.array([0, 0, 1])
            g_rotated = g_rotated / np.linalg.norm(g_rotated) if np.linalg.norm(g_rotated) > 1e-8 else np.array([0.0, 0.0, 0.0])
            
            comp_centripetal = -Rsphere * np.cross(wdot, g_rotated) + np.cross(wdot, ur) + np.cross(gyr[i, 1:4], g_cross_r)

            # Compensation terms

            comp_term_x = -comp_centripetal[0]
            comp_term_y = -comp_centripetal[1]
            comp_term_z = -comp_centripetal[2]

            comp_x.append(comp_term_x / 9.80665)
            comp_y.append(comp_term_y / 9.80665)
            comp_z.append(comp_term_z / 9.80665)

            # Compensate centripetal 
            ax = ax - comp_term_x
            ay = ay - comp_term_y
            az = az - comp_term_z
            
        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        # Adaptive gain
        factorX = min((0.5 * gx) ** 2, 1) # factor is 1 at most
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

        q = np.array([q0, q1, q2, q3])

        # Complementary filter step
        acc_roll = np.arctan2(ayNom, azNom)
        acc_pitch = np.arctan2(-axNom, np.sqrt(ayNom * ayNom + azNom * azNom))

        # Get RPY from quaternion
        try:
            rotation = R.from_quat([q1, q2, q3, q0])
            roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
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

    # Plot compensation terms if any data was collected
    if comp_x and comp_y and comp_z:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(comp_x, label='Compensation X')
        ax.plot(comp_y, label='Compensation Y')
        ax.plot(comp_z, label='Compensation Z')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Compensation Term Value')
        ax.set_title('IMU Compensation Terms')
        ax.legend()
        ax.grid(True)
        #plt.show()

    return quats