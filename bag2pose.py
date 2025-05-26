import numpy as np
import rosbag 
from utils import _quaternion_to_eulerXYZ

def load_rosbag_poses(bag_path, topic, tstart=0.0, tend=None):
    """
    Loads poses from a ROS bag file and returns them as a numpy array:
    [timestamp, x, y, z, yaw, pitch, roll] (angles in degrees).
    Only returns poses between tstart and tend (in seconds, relative to first timestamp).
    Parameters:
        bag_path: Path to the .bag file.
        topic: Name of the topic containing geometry_msgs/Pose or PoseStamped messages.
        tstart: Start time in seconds (relative to first timestamp).
        tend: End time in seconds (relative to first timestamp). If None, goes to end.
    Returns:
        poses: numpy array of shape (N, 7)
    """
    poses = []
    with rosbag.Bag(bag_path, 'r') as bag:
        t0 = None
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            # Handle PoseStamped or Pose
            if hasattr(msg, 'pose'):
                pose = msg.pose
                if hasattr(pose, 'pose'):
                    pose = pose.pose
            else:
                pose = msg

            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            q = pose.orientation
            # Convert quaternion to euler angles (yaw, pitch, roll)
            roll, pitch, yaw = _quaternion_to_eulerXYZ(q.x, q.y, q.z, q.w)
            # Convert to degrees
            yaw = np.rad2deg(yaw)
            pitch = np.rad2deg(pitch)
            roll = np.rad2deg(roll)
            # Use ROS time in seconds
            timestamp = t.to_sec()
            if t0 is None:
                t0 = timestamp
            timestamp -= t0
            poses.append([timestamp, x, y, z, yaw, pitch, roll])
    poses = np.array(poses)
    # Filter by tstart and tend
    if tend is None:
        tend = poses[-1, 0]
    mask = (poses[:, 0] >= tstart) & (poses[:, 0] <= tend)
    return poses[mask]