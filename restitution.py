"""Calculate the restitution coefficient of the puck"""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import puck_data_2

PROJECT_PATH = str(Path(__file__).resolve().parents[0])


def calculate_restitution():
    """Calculate the restitution coefficient by comparing kinetic energy before and after wall collisions."""
    # Load the puck data
    csv_path = f"{PROJECT_PATH}/data/position-mar-11-1.csv"
    data = pd.read_csv(csv_path)
    
    # Segment the data to identify wall collisions
    segments = puck_data_2.segment_puck_trajectory(csv_path, accel_threshold=1, buffer=1)
    
    restitution_values = []
    pre_ke_values = []
    
    for i in range(len(segments)-1):
        # Extract pre and post collision velocities
        #last two columns are x and y displacement - need 2 x vals and 2 y vals
        try:
            precollision = segments[i]
            postcollision = segments[i+1]
            x00, y00, _ = precollision[-2]
            x01, y01, t0 = precollision[-1]
            x10, y10, _ = postcollision[0]
            x11, y11, t1 = postcollision[1]
        except IndexError:
            print("Error: segment index out of range.")
            continue

        pre_velocity = np.array([(x01 - x00) / t0, (y01 - y00) / t0])
        print(pre_velocity)
        post_velocity = np.array([(x11 - x10) / t1, (y11 - y10) / t1])
        print(post_velocity)
        
        # Calculate kinetic energy before and after (proportional to velocity squared)
        pre_ke = np.sum(np.square(pre_velocity))
        post_ke = np.sum(np.square(post_velocity))
        
        # Calculate restitution coefficient (sqrt of energy ratio)
        if pre_ke > 0:  # Avoid division by zero
            restitution = np.sqrt(post_ke / pre_ke)
            if restitution > 1.2:
                print(f"Warning: restitution coefficient {restitution:.4f}: mallet hit?")
                continue
            elif restitution < 0.2:
                print(f"Warning: low restitution coefficient {restitution:.4f}: mallet hit?")
                continue
            elif pre_ke > 20:
                print(f"Warning: high pre-collision energy {pre_ke:.4f}: tracking error?")
                continue
            print(restitution)
            restitution_values.append(restitution)
            pre_ke_values.append(pre_ke)
    
    return (restitution_values, pre_ke_values)

def analyze_results(restitution_values):
    """Analyze and visualize the restitution coefficient results."""
    if not restitution_values:
        print("No valid collisions found.")
        return
    
    # Calculate statistics
    mean_restitution = np.mean(restitution_values)
    median_restitution = np.median(restitution_values)
    std_dev = np.std(restitution_values)
    
    print(f"Restitution Coefficient Analysis:")
    print(f"Number of collisions analyzed: {len(restitution_values)}")
    print(f"Mean restitution: {mean_restitution:.4f}")
    print(f"Median restitution: {median_restitution:.4f}")
    print(f"Standard deviation: {std_dev:.4f}")

def main():
    """Main function to execute the analysis."""
    restitution_values, pre_ke_vals = calculate_restitution()
    analyze_results(restitution_values)
    plt.scatter(pre_ke_vals,restitution_values)
    plt.ylabel("Restitution Coefficient")
    plt.xlabel("Pre-collision Kinetic Energy m/s^2")
    plt.show()

if __name__ == "__main__":
    main()