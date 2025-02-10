import numpy as np
import pandas as pd
from pathlib import Path

def segment_puck_trajectory(x, y, t, velocity_threshold=2.0, buffer=3):
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
    speed = np.sqrt(vx**2 + vy**2)
    
    # Find sudden velocity changes (collisions)
    accelx = np.abs(np.diff(vx))
    accely = np.abs(np.diff(vy))
    collision_indices = np.where(accelx > velocity_threshold or accely > velocity_threshold)[0] + 1
    
    # Add buffer region
    split_indices = set()
    for idx in collision_indices:
        split_indices.update(range(max(0, idx - buffer), min(len(x) - 1, idx + buffer)))
    split_indices = sorted(split_indices)
    
    # Split trajectory into segments
    segments = []

    start = 0
    for idx in split_indices:
        if idx > start:
            segments.append((x[start:idx], y[start:idx], t[start:idx]))
        start = idx + 1
    if start < len(x):
        segments.append((x[start:], y[start:], t[start:]))
    
    # Compute distance traveled for each segment
    distance_segments = []
    for x_seg, y_seg in segments:
        dx = np.diff(x_seg, prepend=x_seg[0])
        dy = np.diff(y_seg, prepend=y_seg[0])
        distances = np.sqrt(dx**2 + dy**2)
        distance_segments.append(distances)
    
    return distance_segments

PROJECT_PATH = str(Path(__file__).resolve().parents[0])

data = pd.read_csv(PROJECT_PATH + '/data/position_13.csv')
print("a")

distance_sesegments = segment_puck_trajectory(data['x'], data['y'], data['dt'], velocity_threshold=2.0, buffer=3)
print("a")