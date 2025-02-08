"""method to visualize the data"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

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

def segment_data(datacsv) -> pd.DataFrame:
    """Segment the data into different sections, based on when the puck hits the walls"""
    #a rapid change x or y velocity indicates a collision, segment based on this
    data = pd.read_csv(datacsv)
    data['vx'] = data['x'].diff() / data['dt']
    data['vy'] = data['y'].diff() / data['dt']
    data['ax'] = data['vx'].diff() / data['dt']
    data['ay'] = data['vy'].diff() / data['dt']
    # an acceleration of abs(5) is a good indicator of a collision
    data['collision'] = (abs(data['ax']) > 5) | (abs(data['ay']) > 5)
    data['collision'] = data['collision'].cumsum()
    return data

def visualize_segments(datacsv):
    """Visualize the data from the csv file
    Segments are separated by different colors"""
    data = segment_data(datacsv)
    x = data['x']
    y = data['y']
    plt.scatter(x, y, c=data['collision'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Puck Trajectory')
    plt.show()

if __name__ == "__main__":
    visualize_segments(f"{PROJECT_PATH}/data/position_13.csv")
