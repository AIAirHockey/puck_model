import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import puck_data_2

PROJECT_PATH = str(Path(__file__).resolve().parents[0])

def calculate_restitution():
    """Calculate the normal and parallel restitution coefficients for horizontal wall collisions."""
    # Load the puck data
    csv_path = f"{PROJECT_PATH}/data/position-mar-14-1.csv"
    data = pd.read_csv(csv_path)
    
    # Segment the data to identify wall collisions
    segments, _ = puck_data_2.segment_puck_trajectory(csv_path, accel_threshold=1, buffer=1)
    
    # Separate data for different walls
    bottom_wall_data = {
        'normal_restitution': [],
        'parallel_restitution': [],
        'pre_ve_values': [],
        'x_direction_change': []
    }
    
    top_wall_data = {
        'normal_restitution': [],
        'parallel_restitution': [],
        'pre_ve_values': [],
        'x_direction_change': []
    }

    combined_data = {
        'normal_restitution': [],
        'parallel_restitution': [],
        'pre_ve_values': [],
        'x_direction_change': []
    }
    
    # For plotting which walls were hit
    collision_points = []
    wall_types = []
    
    for i in range(len(segments)-1):
        try:
            precollision = segments[i]
            postcollision = segments[i+1]
            
            # Get collision point (last point of pre-collision segment)
            collision_x, collision_y, _ = precollision[-1]
            
            # Skip collisions near corner walls
            if collision_x < 0.1 or collision_x > 1.9:
                continue
            if collision_y > 0.1 and collision_y < 0.9:
                print("rejected due to y coordinate")
                continue
                
            # Extract velocities before and after collision
            x00, y00, _ = precollision[-2]
            x01, y01, t0 = precollision[-1]
            x10, y10, _ = postcollision[0]
            x11, y11, t1 = postcollision[1]
            
            # Calculate pre and post velocities
            pre_vx = (x01 - x00) / t0
            pre_vy = (y01 - y00) / t0
            post_vx = (x11 - x10) / t1
            post_vy = (y11 - y10) / t1
            
            # Detect which wall was hit based on y-coordinate and y-velocity
            wall_hit = None
            
            # Bottom wall (y≈0) -> negative y-velocity will flip to positive
            if collision_y < 0.1 and pre_vy < 0 and post_vy > 0:
                wall_hit = "bottom"
            # Top wall (y≈1) -> positive y-velocity will flip to negative
            elif collision_y > 0.9 and pre_vy > 0 and post_vy < 0:
                wall_hit = "top"
            else:
                # Not a clear horizontal wall collision
                continue
            
            # Calculate normal restitution (y-component)
            normal_restitution = abs(post_vy / pre_vy) if abs(pre_vy) > 0.1 else None

            # Check if x-direction changed
            x_direction_changed = (pre_vx * post_vx) < 0
            
            # Calculate parallel restitution (x-component)
            if not x_direction_changed:
                parallel_restitution = abs(post_vx / pre_vx) if abs(pre_vx) > 0.1 else None
            else:
                parallel_restitution = None
            
            
            
            # Calculate kinetic energy before collision
            pre_ve = np.sqrt(pre_vx**2 + pre_vy**2)
            
            # Store data based on which wall was hit
            if normal_restitution is not None:
                combined_data['normal_restitution'].append(normal_restitution)
                if parallel_restitution is not None:
                    combined_data['parallel_restitution'].append(parallel_restitution)
                combined_data['pre_ve_values'].append(pre_ve)
                combined_data['x_direction_change'].append(x_direction_changed)


            if wall_hit == "bottom" and normal_restitution is not None:
                if 0.2 < normal_restitution < 1.1 and pre_ve < 15:
                    bottom_wall_data['normal_restitution'].append(normal_restitution)
                    if parallel_restitution is not None:
                        bottom_wall_data['parallel_restitution'].append(parallel_restitution)
                    bottom_wall_data['pre_ve_values'].append(pre_ve)
                    bottom_wall_data['x_direction_change'].append(x_direction_changed)
                    collision_points.append((collision_x, collision_y))
                    wall_types.append('bottom')

            elif wall_hit == "top" and normal_restitution is not None:
                if 0.2 < normal_restitution < 1.1 and pre_ve < 15:
                    top_wall_data['normal_restitution'].append(normal_restitution)
                    if parallel_restitution is not None:
                        top_wall_data['parallel_restitution'].append(parallel_restitution)
                    top_wall_data['pre_ve_values'].append(pre_ve)
                    top_wall_data['x_direction_change'].append(x_direction_changed)
                    collision_points.append((collision_x, collision_y))
                    wall_types.append('top')
                    
            print(f"Wall: {wall_hit}, Normal e: {normal_restitution}, Parallel e: {parallel_restitution}, X-dir change: {x_direction_changed}")
            
        except (IndexError, ZeroDivisionError) as e:
            print(f"Error: {e}")
            continue
    
    return combined_data, bottom_wall_data, top_wall_data, collision_points, wall_types

def analyze_results(bottom_data, top_data, combined_data):
    """Analyze and visualize the restitution coefficient results."""
    
    # Function to print statistics for a dataset
    def print_stats(name, data):
        if not data['normal_restitution']:
            print(f"No valid {name} wall collisions found.")
            return
        
        # Calculate statistics for normal component
        n_mean = np.mean(data['normal_restitution'])
        n_median = np.median(data['normal_restitution'])
        n_std = np.std(data['normal_restitution'])
        
        # Calculate statistics for parallel component
        p_mean = np.mean(data['parallel_restitution']) if data['parallel_restitution'] else 0
        p_median = np.median(data['parallel_restitution']) if data['parallel_restitution'] else 0
        p_std = np.std(data['parallel_restitution']) if data['parallel_restitution'] else 0
        
        # Count direction changes
        dir_changes = sum(data['x_direction_change'])
        
        print(f"\n{name.capitalize()} Wall Restitution Analysis:")
        print(f"Number of collisions analyzed: {len(data['normal_restitution'])}")
        print(f"Normal component (y-velocity):")
        print(f"  Mean: {n_mean:.4f}")
        print(f"  Median: {n_median:.4f}")
        print(f"  Standard deviation: {n_std}")
        
        if data['parallel_restitution']:
            print(f"Parallel component (x-velocity):")
            print(f"  Mean: {p_mean:.4f}")
            print(f"  Median: {p_median:.4f}")
            print(f"  Standard deviation: {p_std:.4f}")
            
        print(f"X direction changes: {dir_changes} out of {len(data['x_direction_change'])} collisions ({dir_changes/len(data['x_direction_change'])*100:.1f}%)")
    
    # Print statistics for both walls
    print_stats("bottom", bottom_data)
    print_stats("top", top_data)
    print_stats("combined", combined_data)


def plot_results(bottom_data, top_data, collision_points, wall_types):
    """Plot the results."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Normal Restitution Distribution
    ax1 = fig.add_subplot(231)
    bins = np.linspace(0.2, 1.2, 20)
    if bottom_data['normal_restitution']:
        ax1.hist(bottom_data['normal_restitution'], bins=bins, alpha=0.5, label='Bottom Wall')
    if top_data['normal_restitution']:
        ax1.hist(top_data['normal_restitution'], bins=bins, alpha=0.5, label='Top Wall')
    ax1.set_xlabel('Normal Restitution Coefficient')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Normal Restitution Coefficients')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parallel Restitution Distribution
    ax2 = fig.add_subplot(232)
    bins = np.linspace(0.5, 1.5, 20)
    if bottom_data['parallel_restitution']:
        ax2.hist(bottom_data['parallel_restitution'], bins=bins, alpha=0.5, label='Bottom Wall')
    if top_data['parallel_restitution']:
        ax2.hist(top_data['parallel_restitution'], bins=bins, alpha=0.5, label='Top Wall')
    ax2.set_xlabel('Parallel Restitution Coefficient')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Parallel Restitution Coefficients')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Normal vs Parallel Restitution
    ax3 = fig.add_subplot(233)
    
    # For bottom wall, create paired data points only where both normal and parallel exist
    if bottom_data['normal_restitution'] and bottom_data['parallel_restitution']:
        # Create index lists for each collision that has both values
        bottom_parallel_idx = []
        bottom_normal_values = []
        bottom_parallel_values = []
        
        for i, is_change in enumerate(bottom_data['x_direction_change']):
            if i < len(bottom_data['normal_restitution']):
                if not is_change and i < len(bottom_data['parallel_restitution']):
                    bottom_normal_values.append(bottom_data['normal_restitution'][i])
                    bottom_parallel_values.append(bottom_data['parallel_restitution'][len(bottom_parallel_idx)])
                    bottom_parallel_idx.append(i)
        
        if bottom_normal_values and bottom_parallel_values:
            ax3.scatter(bottom_normal_values, bottom_parallel_values, 
                      alpha=0.7, label='Bottom Wall', marker='o')
    
    # For top wall, similar approach
    if top_data['normal_restitution'] and top_data['parallel_restitution']:
        # Create index lists for each collision that has both values
        top_parallel_idx = []
        top_normal_values = []
        top_parallel_values = []
        
        for i, is_change in enumerate(top_data['x_direction_change']):
            if i < len(top_data['normal_restitution']):
                if not is_change and i < len(top_data['parallel_restitution']):
                    top_normal_values.append(top_data['normal_restitution'][i])
                    top_parallel_values.append(top_data['parallel_restitution'][len(top_parallel_idx)])
                    top_parallel_idx.append(i)
        
        if top_normal_values and top_parallel_values:
            ax3.scatter(top_normal_values, top_parallel_values, 
                      alpha=0.7, label='Top Wall', marker='x')
    
    ax3.set_xlabel('Normal Restitution')
    ax3.set_ylabel('Parallel Restitution')
    ax3.set_title('Normal vs. Parallel Restitution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Restitution vs Velocity
    ax4 = fig.add_subplot(234)
    if bottom_data['normal_restitution']:
        ax4.scatter(np.sqrt(bottom_data['pre_ve_values']), bottom_data['normal_restitution'], 
                   alpha=0.7, label='Bottom Wall', marker='o')
    if top_data['normal_restitution']:
        ax4.scatter(np.sqrt(top_data['pre_ve_values']), top_data['normal_restitution'], 
                   alpha=0.7, label='Top Wall', marker='x')
    ax4.set_xlabel('Pre-collision Velocity (m/s)')
    ax4.set_ylabel('Normal Restitution')
    ax4.set_title('Normal Restitution vs. Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Collision Locations
    ax5 = fig.add_subplot(235)
    for i, (x, y) in enumerate(collision_points):
        color = 'blue' if wall_types[i] == 'bottom' else 'red'
        ax5.scatter(x, y, color=color, alpha=0.7)
    ax5.set_xlabel('X Position')
    ax5.set_ylabel('Y Position')
    ax5.set_title('Collision Locations')
    ax5.set_xlim(0, 2)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    # 6. X-Direction Changes
    ax6 = fig.add_subplot(236)
    labels = ['No Change', 'Changed']
    bottom_changes = [len(bottom_data['x_direction_change']) - sum(bottom_data['x_direction_change']), 
                     sum(bottom_data['x_direction_change'])]
    top_changes = [len(top_data['x_direction_change']) - sum(top_data['x_direction_change']), 
                  sum(top_data['x_direction_change'])]
    
    x = np.arange(len(labels))
    width = 0.35
    ax6.bar(x - width/2, bottom_changes, width, label='Bottom Wall')
    ax6.bar(x + width/2, top_changes, width, label='Top Wall')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.set_ylabel('Count')
    ax6.set_title('X-Velocity Direction Changes')
    ax6.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to execute the analysis."""
    combined_data, bottom_data, top_data, collision_points, wall_types = calculate_restitution()
    analyze_results(bottom_data, top_data, combined_data)
    plot_results(bottom_data, top_data, collision_points, wall_types)

if __name__ == "__main__":
    main()