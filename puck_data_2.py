import numpy as np
import pandas as pd
from pathlib import Path

def segment_puck_trajectory(x, y, t, accel_threshold=2.0, buffer=3):
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
            trajectories.append(trajectory)
            trajectory = []
    trajectories.append(trajectory)
    trajectory = []

    return trajectories

PROJECT_PATH = str(Path(__file__).resolve().parents[0])

data = pd.read_csv(PROJECT_PATH + '/data/position_13.csv')
print("a")

distance_sesegments = segment_puck_trajectory(data['x'], data['y'], data['dt'], accel_threshold=2.0, buffer=3)
print("a")