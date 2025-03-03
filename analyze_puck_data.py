"""method to visualize the data"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_PATH = str(Path(__file__).resolve().parents[0])


def position_plot(datacsv):
    """Visualize the data from the csv file"""
    data = pd.read_csv(datacsv)
    x = data['x']
    y = data['y']
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Puck Trajectory')
    plt.show()

def velocity_plot(datacsv):
    """use the dt to calculate the velocity at each time step and plot it"""
    data = pd.read_csv(datacsv)

    # Calculate the velocities
    data['vx'] = data['x'].diff() / data['dt']
    data['vy'] = data['y'].diff() / data['dt']

    # Calculate the time
    data['time'] = data['dt'].cumsum()

    # Plot the velocities
    plt.figure(figsize=(10, 5))
    plt.plot(data['time'], data['vx'], label='x velocity')
    plt.plot(data['time'], data['vy'], label='y velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def acceleration_plot(datacsv):
    """use the dt to calculate the acceleration at each time step and plot it"""
    data = pd.read_csv(datacsv)

    # Calculate the velocities
    data['vx'] = data['x'].diff() / data['dt']
    data['vy'] = data['y'].diff() / data['dt']

    # Calculate the accelerations
    data['ax'] = data['vx'].diff() / data['dt']
    data['ay'] = data['vy'].diff() / data['dt']

    # Calculate the time
    data['time'] = data['dt'].cumsum()

    # Plot the accelerations
    plt.figure(figsize=(10, 5))
    plt.plot(data['time'], data['ax'], label='x acceleration')
    plt.plot(data['time'], data['ay'], label='y acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Acceleration over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def segment_data(datacsv, accel_threshold=1.0, buffer=0):
    """
    Segments the puck trajectory based on sudden velocity changes and computes distance traveled.
    
    Parameters:
    x (array-like): X-coordinates of the puck.
    y (array-like): Y-coordinates of the puck.
    t (array-like): Time vector of dts corresponding to x and y.
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
            trajectories.append(trajectory)
            trajectory = []
    trajectories.append(trajectory)
    trajectory = []
    
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

def visualize_segments(datacsv):
    """Visualize the data from the csv file
    Segments are separated by different colors"""
    trajectories, collisions = segment_data(datacsv)
    x = np.concatenate([np.array([point[0] for point in traj]) for traj in trajectories])
    y = np.concatenate([np.array([point[1] for point in traj]) for traj in trajectories])
    plt.scatter(x, y, c=collisions)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Puck Trajectory')
    plt.show()

if __name__ == "__main__":
    position_plot(f"{PROJECT_PATH}/data/position_9.csv")
    visualize_segments(f"{PROJECT_PATH}/data/position_9.csv")
