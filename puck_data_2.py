import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

def segment_puck_trajectory(datacsv, accel_threshold=0.5, buffer=3):
    """
    Segments the puck trajectory based on sudden velocity changes and computes distance traveled.
    
    Parameters:
    x (array-like): X-coordinates of the puck.
    y (array-like): Y-coordinates of the puck.
    t (array-like): Time vector corresponding to x and y.
    velocity_threshold (float): Threshold for detecting sudden velocity changes.
    buffer (int): Number of points to ignore around collisions.
    
    Returns:
    list of np.ndarray: Segmented distance traveled arrays.
    """
    data = pd.read_csv(datacsv)
    x = data['x']
    y = data['y']
    t = data['dt']
    x, y, t = np.array(x), np.array(y), np.array(t)
    
    # Compute finite difference velocities
    vx = np.diff(x) / t[1:]
    vy = np.diff(y) / t[1:]
    
    # Find sudden velocity changes (collisions)
    accelx = np.abs(np.diff(vx))
    accely = np.abs(np.diff(vy))
    collision_indices = np.where(np.logical_or(accelx > accel_threshold, accely > accel_threshold))[0] + 1
    collision_indices_buffer = set()
    for idx in collision_indices:
        collision_indices_buffer.update(range(max(0, idx - buffer), min(len(x), idx + buffer + 1)))

    trajectories = []
    trajectory = []
    for i in range(len(x)):
        if i not in collision_indices_buffer:
            trajectory.append((x[i], y[i], t[i]))
        elif i in collision_indices_buffer and i-1 not in collision_indices_buffer:
            if len(trajectory) > 1:
                trajectories.append(np.array(trajectory))
            trajectory = []
    if len(trajectory) > 1:
        trajectories.append(np.array(trajectory))

    collisions = np.zeros(sum(len(traj) for traj in trajectories))
    collision_idx = 0
    i = 0
    for traj in trajectories:
        if i < len(trajectories) - 1:
            collision_idx += len(traj)
            collisions[collision_idx] = 1
            i += 1
    collisions = np.cumsum(collisions)

    return trajectories, collisions


def compute_v0(trajectory):
    """
    Computes the initial velocity from a trajectory segment.
    
    Parameters:
    trajectory (np.ndarray): Trajectory segment with columns [x, y, t].
    
    Returns:
    float: Initial velocity.
    """
    x0, y0, _ = trajectory[0]
    x1, y1, t1 = trajectory[1]
    return np.sqrt((x1 - x0)**2 + (y1 - y0)**2) / (t1)

def get_time_array(trajectory):
    """
    Extracts the time array from a trajectory segment. And turns it into an elapsed time array.
    (ie like linspace)
    
    Parameters:
    trajectory (np.ndarray): Trajectory segment with columns [x, y, t].
    
    Returns:
    np.ndarray: Cumulative Time array.
    """
    dts = np.array([point[2] for point in trajectory])
    return np.cumsum(dts)

def plot_segments(segments):
    """
    Plots the trajectory segments.
    
    Parameters:
    segments (list of np.ndarray): List of trajectory segments.
    """
    plt.figure(figsize=(10, 6))
    for segment in segments:
        try:
            plt.plot(segment[:,0], segment[:,1], 'o-')
        except IndexError:
            continue
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Puck Trajectory Segments")
    plt.legend(range(len(segments)))
    plt.show()


if __name__ == "__main__":
    PROJECT_PATH = str(Path(__file__).resolve().parents[0])

    data = PROJECT_PATH + '/data/position-mar-11-2.csv'
    distance_segments = segment_puck_trajectory(data, accel_threshold=1, buffer=1)
    plot_segments(distance_segments)
    
