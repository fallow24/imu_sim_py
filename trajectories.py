from scipy.spatial.transform import Rotation as Rot
import numpy as np

def generate_trochoid_forward(duration, fs, R, r, omega):
    """
    Generates a sample IMU trajectory resembling a trochoid in 3D space.
    The IMU is rigidly mounted inside a ball that is rolling on flat floor without slippage.
    Parameters are:
        duration: How long the trajectory should be in seconds.
        fs: How many samples 1 second should contain.
        R: Radius of the rolling ball.
        r: Offset from ball center.
        omega: Rotation speed of the ball.
    """
    N = int(duration * fs)
    timestamps = np.linspace(0, duration, N)

    # Trochoid equations for a ball rolling along x-y plane
    theta = omega * timestamps
    x = R * theta - r * np.sin(theta)
    y = 0 * theta  # stays on the floor
    z = R - r * np.cos(theta)

    # Orientation: yaw follows the pitching direction
    yaw = np.zeros_like(theta)
    pitch = np.rad2deg(theta) % 360
    roll = np.zeros_like(theta)

    # Stack into poses: [timestamp, x, y, z, yaw, pitch, roll]
    poses = np.column_stack((timestamps, x, y, z, yaw, pitch, roll))
    return poses

def generate_trochoid_curved(duration, fs, R, r, omega, turn_period=5.0, turn_angle_deg=45):
    """
    Generates a trochoid trajectory for a ball rolling on a floor, but with periodic left/right turns.
    The ball's heading (yaw) oscillates sinusoidally, causing the ball to curve left and right as it rolls.
    Parameters:
        duration: Duration in seconds.
        fs: Samples per second.
        R: Radius of the rolling ball.
        r: Offset from ball center.
        omega: Rolling angular velocity (rad/s).
        turn_period: Period (s) of a full left-right-left turn cycle.
        turn_angle_deg: Maximum yaw angle left/right (degrees).
    Returns:
        poses: [timestamp, x, y, z, yaw, pitch, roll]
    """
    N = int(duration * fs)
    timestamps = np.linspace(0, duration, N)

    # Yaw oscillates sinusoidally to simulate left/right turns
    turn_angle_rad = np.deg2rad(turn_angle_deg)
    yaw = turn_angle_rad * np.sin(2 * np.pi * timestamps / turn_period)  # radians

    # Integrate yaw rate to get heading at each time step
    heading = np.cumsum(np.gradient(yaw, timestamps)) / fs  # radians

    # Trochoid equations, but now the heading changes
    theta = omega * timestamps
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(1, N):
        # Step forward in the current heading direction
        dx = (R * (theta[i] - theta[i-1]) - r * (np.sin(theta[i]) - np.sin(theta[i-1]))) * np.cos(heading[i])
        dy = (R * (theta[i] - theta[i-1]) - r * (np.sin(theta[i]) - np.sin(theta[i-1]))) * np.sin(heading[i])
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
    z = R - r * np.cos(theta)

    # Orientation: yaw follows the heading, pitch follows theta, roll is zero
    yaw_deg = np.rad2deg(heading)
    pitch = np.rad2deg(theta) % 360
    roll = np.zeros_like(theta)

    poses = np.column_stack((timestamps, x, y, z, yaw_deg, pitch, roll))
    return poses

def generate_trochoid(omega_vec, R_ball, offset_vec):
    """
    Generates a trochoid trajectory for a ball rolling on a floor, with arbitrary angular velocity at each time step,
    and allows for an arbitrary 3D offset of the IMU sensor from the ball center.
    Parameters:
        omega_vec: (N, 4) numpy array, first column is timestamp, next three columns are angular velocity vector [wx, wy, wz] at each time step (rad/s).
        R_ball: Radius of the rolling ball.
        offset_vec: 3D numpy array, offset of the IMU from the ball center [x, y, z] in the ball's local frame.
    Returns:
        poses: [timestamp, x, y, z, yaw, pitch, roll]
    """
    omega_vec = np.asarray(omega_vec)
    assert omega_vec.shape[1] == 4, "omega_vec must have shape (N, 4): [timestamp, wx, wy, wz]"
    timestamps = omega_vec[:, 0]
    N = len(timestamps)
    dt = np.diff(timestamps, prepend=timestamps[0])

    rotations = []
    rot = Rot.identity()
    for i in range(N):
        omega = omega_vec[i, 1:4]
        angle = np.linalg.norm(omega) * dt[i]
        if angle > 1e-12:
            axis = omega / np.linalg.norm(omega)
            dR = Rot.from_rotvec(axis * angle)
            rot = rot * dR
        rotations.append(rot)

    # Integrate translational velocity (rolling constraint)
    positions = np.zeros((N, 3))
    for i in range(1, N):
        # The contact point is always at -R_ball * (current z-axis in world frame)
        z_axis_world = np.array([0,0,1])#rotations[i-1].apply([0, 0, 1])
        r_contact = -R_ball * z_axis_world
        omega = omega_vec[i-1, 1:4]
        v_center = np.cross(omega, r_contact)
        positions[i] = positions[i-1] + v_center * dt[i]

    # Compute IMU position by applying offset_vec in local frame to each orientation
    imu_positions = np.zeros((N, 3))
    yaws = np.zeros(N)
    pitches = np.zeros(N)
    rolls = np.zeros(N)
    for i in range(N):
        offset_global = rotations[i].apply(offset_vec)
        imu_positions[i] = positions[i] + offset_global
        yaw, pitch, roll = rotations[i].as_euler('zyx', degrees=True)
        yaws[i] = yaw
        pitches[i] = pitch
        rolls[i] = roll

    poses = np.column_stack((timestamps, imu_positions[:,0], imu_positions[:,1], imu_positions[:,2], yaws, pitches, rolls))
    return poses

def generate_rotating_disc(duration, fs, imu_offset, omega):
    """
    Generates a sample IMU trajectory for an IMU mounted on a rotating disc.
    The disc rotates in place (center does not move), IMU is offset from center.
    Parameters:
        duration: How long the trajectory should be in seconds.
        fs: How many samples per second.
        imu_offset: Offset of IMU from disc center.
        omega: Angular velocity of the disc (rad/s).
    """
    N = int(duration * fs)
    timestamps = np.linspace(0, duration, N)

    theta = omega * timestamps
    x = imu_offset * np.cos(theta)
    y = imu_offset * np.sin(theta)
    z = np.zeros_like(theta)

    # Orientation: yaw follows the rotation, pitch and roll are zero
    yaw = np.rad2deg(theta) % 360
    pitch = np.zeros_like(theta)
    roll = np.zeros_like(theta)

    poses = np.column_stack((timestamps, x, y, z, yaw, pitch, roll))
    return poses

def generate_lift_motion(duration, fs, lift_height):
    """
    Generates a sample IMU trajectory for a lift (elevator) moving vertically with smooth transitions.
    The lift starts at rest, ramps up acceleration, moves at constant acceleration, ramps down to constant velocity,
    then ramps up negative acceleration, and finally ramps down to rest.
    Parameters:
        duration: Total duration of the motion (seconds).
        fs: Samples per second.
        lift_height: Total vertical distance to travel (meters).
    Returns:
        poses: [timestamp, x, y, z, yaw, pitch, roll]
    """
    N = int(duration * fs)
    timestamps = np.linspace(0, duration, N)

    n_rest = 5
    n_motion = N - 2 * n_rest

    # Define ramp durations (as fractions of motion)
    ramp_frac = 0.1  # 10% for each ramp up/down
    ramp_n = int(ramp_frac * n_motion)
    const_accel_n = int(0.1 * n_motion)  # 10% constant accel
    const_vel_n = n_motion - 4 * ramp_n - 2 * const_accel_n  # rest is constant velocity

    # Build time arrays for each phase
    t_ramp = np.linspace(0, ramp_n / fs, ramp_n, endpoint=False)
    t_const_accel = np.linspace(0, const_accel_n / fs, const_accel_n, endpoint=False)
    t_const_vel = np.linspace(0, const_vel_n / fs, const_vel_n, endpoint=False)

    def compute_displacement(a_max):
        # Forward motion
        # Ramp up
        s1 = (1/6) * a_max * (t_ramp[-1] + 1/fs)**3 / (t_ramp[-1] + 1/fs)
        # Const accel
        s2 = 0.5 * a_max * (t_const_accel[-1] + 1/fs)**2
        # Ramp down
        s3 = (1/6) * a_max * (t_ramp[-1] + 1/fs)**3 / (t_ramp[-1] + 1/fs)
        # Constant velocity
        v_peak = a_max * (t_ramp[-1] + 1/fs) + a_max * (t_const_accel[-1] + 1/fs)
        s4 = v_peak * (t_const_vel[-1] + 1/fs)
        # Backward motion (mirror)
        return 2 * (s1 + s2 + s3) + s4

    # Use a simple search to find a_max
    a_max = 1.0
    for _ in range(20):
        disp = compute_displacement(a_max)
        a_max *= lift_height / disp

    # Now build the acceleration profile
    acc_profile = []
    # Ramp up
    acc_profile += list(np.linspace(0, a_max, ramp_n, endpoint=False))
    # Const accel
    acc_profile += [a_max] * const_accel_n
    # Ramp down
    acc_profile += list(np.linspace(a_max, 0, ramp_n, endpoint=False))
    # Const vel
    acc_profile += [0] * const_vel_n
    # Ramp up (neg)
    acc_profile += list(np.linspace(0, -a_max, ramp_n, endpoint=False))
    # Const decel
    acc_profile += [-a_max] * const_accel_n
    # Ramp down (neg)
    acc_profile += list(np.linspace(-a_max, 0, ramp_n, endpoint=False))

    acc_profile = np.array(acc_profile)
    # Pad if needed
    if len(acc_profile) < n_motion:
        acc_profile = np.append(acc_profile, [0] * (n_motion - len(acc_profile)))
    elif len(acc_profile) > n_motion:
        acc_profile = acc_profile[:n_motion]

    # Integrate acceleration to get velocity and position
    vel_profile = np.cumsum(acc_profile) / fs
    pos_profile = np.cumsum(vel_profile) / fs

    # Initial and final rest
    z_rest_start = np.zeros(n_rest)
    z_rest_end = np.ones(n_rest) * pos_profile[-1]

    z = np.concatenate([z_rest_start, pos_profile, z_rest_end])
    if len(z) < N:
        z = np.append(z, [z[-1]] * (N - len(z)))
    elif len(z) > N:
        z = z[:N]

    x = np.zeros_like(z)
    y = np.zeros_like(z)
    yaw = np.zeros_like(z)
    pitch = np.zeros_like(z)
    roll = np.zeros_like(z)

    poses = np.column_stack((timestamps, x, y, z, yaw, pitch, roll))
    return poses

def generate_stationary_pose(duration, fs, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0):
    """
    Generates a stationary IMU trajectory (no movement, no rotation).
    Parameters:
        duration: Duration in seconds.
        fs: Sampling frequency (Hz).
        x, y, z: Fixed position coordinates.
        yaw, pitch, roll: Fixed orientation (degrees).
    Returns:
        poses: [timestamp, x, y, z, yaw, pitch, roll]
    """
    N = int(duration * fs)
    timestamps = np.linspace(0, duration, N)
    poses = np.column_stack((
        timestamps,
        np.full(N, x),
        np.full(N, y),
        np.full(N, z),
        np.full(N, yaw),
        np.full(N, pitch),
        np.full(N, roll)
    ))
    return poses