import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from scipy.stats import circmean
import math
from puck_data_2 import segment_puck_trajectory

def analyze_collision_angles(segments, wall_threshold=10, visualize=True, segment_indices=None):
    """
    Analyzes collision angles from segmented puck trajectory data.
    
    Parameters:
    segments (list of np.ndarray): List of trajectory segments before and after collisions.
    wall_threshold (float): Distance from edge to consider as wall collision (in same units as x,y).
    visualize (bool): Whether to visualize the collisions and angles.
    segment_indices (list): Specific segment indices to analyze (None = all segments).
    
    Returns:
    dict: Dictionary containing incidence and reflection angles for each wall.
    """
    # Determine table boundaries (assuming rectangular table)
    all_x = np.concatenate([seg[:,0] for seg in segments])
    all_y = np.concatenate([seg[:,1] for seg in segments])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # Add a small margin for better visualization
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    table_bounds = {
        'left': x_min + wall_threshold,
        'right': x_max - wall_threshold,
        'bottom': y_min + wall_threshold,
        'top': y_max - wall_threshold
    }
    
    # For storing angles by wall
    collision_angles = {
        'left': {'incidence': [], 'reflection': []},
        'right': {'incidence': [], 'reflection': []},
        'bottom': {'incidence': [], 'reflection': []},
        'top': {'incidence': [], 'reflection': []}
    }
    
    # For visualization
    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        ax_traj = axs[0]
        ax_angles = axs[1]
        
        # Plot table bounds
        ax_traj.add_patch(Rectangle((x_min - x_margin, y_min - y_margin),
                                   (x_max - x_min) + 2*x_margin,
                                   (y_max - y_min) + 2*y_margin,
                                   fill=False, edgecolor='black', linewidth=2))
        
        # Draw walls
        table_walls = {
            'left': ([x_min, x_min], [y_min, y_max], 'b-'),
            'right': ([x_max, x_max], [y_min, y_max], 'b-'),
            'bottom': ([x_min, x_max], [y_min, y_min], 'b-'),
            'top': ([x_min, x_max], [y_max, y_max], 'b-')
        }
        for wall, (x, y, fmt) in table_walls.items():
            ax_traj.plot(x, y, fmt, linewidth=2, label=f'{wall} wall')
    
    # Process segments
    indices_to_process = range(len(segments)-1) if segment_indices is None else segment_indices
    
    for i in indices_to_process:
        if i+1 >= len(segments):
            continue
            
        # Get pre and post collision segments
        pre_segment = segments[i]
        post_segment = segments[i+1]
        
        if len(pre_segment) < 2 or len(post_segment) < 2:
            continue
            
        # Get the last points from pre-collision and first points from post-collision
        pre_points = pre_segment[-2:]  # Last two points before collision
        post_points = post_segment[:2]  # First two points after collision
        
        # Calculate velocity vectors
        pre_vel = np.array([pre_points[1][0] - pre_points[0][0], 
                          pre_points[1][1] - pre_points[0][1]])
        post_vel = np.array([post_points[1][0] - post_points[0][0], 
                           post_points[1][1] - post_points[0][1]])
        
        # Normalize velocity vectors
        pre_vel_norm = pre_vel / np.linalg.norm(pre_vel)
        post_vel_norm = post_vel / np.linalg.norm(post_vel)
        
        # Determine which wall was hit
        collision_point = pre_points[1]  # Last point before collision
        wall_hit = None
        
        if abs(collision_point[0] - x_min) < wall_threshold:
            wall_hit = 'left'
            # Wall normal is (1, 0) for left wall
            wall_normal = np.array([1, 0])
        elif abs(collision_point[0] - x_max) < wall_threshold:
            wall_hit = 'right'
            # Wall normal is (-1, 0) for right wall
            wall_normal = np.array([-1, 0])
        elif abs(collision_point[1] - y_min) < wall_threshold:
            wall_hit = 'bottom'
            # Wall normal is (0, 1) for bottom wall
            wall_normal = np.array([0, 1])
        elif abs(collision_point[1] - y_max) < wall_threshold:
            wall_hit = 'top'
            # Wall normal is (0, -1) for top wall
            wall_normal = np.array([0, -1])
        else:
            # Not a wall collision, might be a mallet hit or other event
            continue
            
        # Calculate angles (in radians)
        # Angle of incidence: angle between incoming vector and wall normal
        dot_product = np.dot(-pre_vel_norm, wall_normal)
        incidence_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Angle of reflection: angle between outgoing vector and wall normal
        dot_product = np.dot(post_vel_norm, wall_normal)
        reflection_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Convert to degrees
        incidence_angle_deg = np.degrees(incidence_angle)
        reflection_angle_deg = np.degrees(reflection_angle)
        
        # Store angles
        collision_angles[wall_hit]['incidence'].append(incidence_angle_deg)
        collision_angles[wall_hit]['reflection'].append(reflection_angle_deg)
        
        if visualize:
            # Plot segments
            ax_traj.plot(pre_segment[:,0], pre_segment[:,1], 'r-', alpha=0.5)
            ax_traj.plot(post_segment[:,0], post_segment[:,1], 'g-', alpha=0.5)
            
            # Plot collision point
            ax_traj.scatter(collision_point[0], collision_point[1], color='b', s=50)
            
            # Plot velocity vectors
            scale = 20  # Scale factor for visualization
            ax_traj.arrow(collision_point[0], collision_point[1], 
                        pre_vel_norm[0]*scale, pre_vel_norm[1]*scale,
                        head_width=2, head_length=3, fc='r', ec='r')
            ax_traj.arrow(collision_point[0], collision_point[1], 
                        post_vel_norm[0]*scale, post_vel_norm[1]*scale,
                        head_width=2, head_length=3, fc='g', ec='g')
            
            # Label with angles
            ax_traj.annotate(f"θi={incidence_angle_deg:.1f}°\nθr={reflection_angle_deg:.1f}°", 
                           (collision_point[0], collision_point[1]),
                           xytext=(10, 10), textcoords='offset points')
    
    # Calculate summary statistics
    summary = {}
    for wall, angles in collision_angles.items():
        if angles['incidence'] and angles['reflection']:
            inc_angles = np.array(angles['incidence'])
            ref_angles = np.array(angles['reflection'])
            error = np.abs(inc_angles - ref_angles)
            
            summary[wall] = {
                'incidence_mean': np.mean(inc_angles),
                'reflection_mean': np.mean(ref_angles),
                'incidence_std': np.std(inc_angles),
                'reflection_std': np.std(ref_angles),
                'mean_error': np.mean(error),
                'count': len(inc_angles)
            }
    
    # Visualization for angle comparison
    if visualize:
        # Configure trajectory plot
        ax_traj.set_title("Puck Trajectory with Collision Angles")
        ax_traj.set_xlabel("X Position")
        ax_traj.set_ylabel("Y Position")
        ax_traj.set_aspect('equal')
        ax_traj.legend(loc='lower right')
        
        # Scatter plot of incidence vs reflection angles
        for wall, angles in collision_angles.items():
            if angles['incidence'] and angles['reflection']:
                ax_angles.scatter(angles['incidence'], angles['reflection'], label=f"{wall} wall")
                
        # Add ideal line (incidence = reflection)
        min_angle = 0
        max_angle = 90
        ax_angles.plot([min_angle, max_angle], [min_angle, max_angle], 'k--', alpha=0.7, label="Perfect reflection")
        
        ax_angles.set_title("Angle of Incidence vs Angle of Reflection")
        ax_angles.set_xlabel("Angle of Incidence (degrees)")
        ax_angles.set_ylabel("Angle of Reflection (degrees)")
        ax_angles.set_xlim(min_angle, max_angle)
        ax_angles.set_ylim(min_angle, max_angle)
        ax_angles.grid(True)
        ax_angles.legend()
        
        # Display summary statistics as text
        textbox_content = "Collision Angle Statistics:\n\n"
        for wall, stats in summary.items():
            if stats['count'] > 0:
                textbox_content += f"{wall.capitalize()} wall ({stats['count']} collisions):\n"
                textbox_content += f"  Incidence: {stats['incidence_mean']:.1f}° ± {stats['incidence_std']:.1f}°\n"
                textbox_content += f"  Reflection: {stats['reflection_mean']:.1f}° ± {stats['reflection_std']:.1f}°\n"
                textbox_content += f"  Mean Error: {stats['mean_error']:.1f}°\n\n"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax_angles.text(1.02, 0.5, textbox_content, transform=ax_angles.transAxes,
                     fontsize=9, verticalalignment='center', bbox=props)
        
        plt.tight_layout()
        plt.show()
    
    return collision_angles, summary

if __name__ == "__main__":
    PROJECT_PATH = str(Path(__file__).resolve().parents[0])
    data = PROJECT_PATH + '/data/position-mar-11-2.csv'
    
    trajectories, _ = segment_puck_trajectory(data, accel_threshold=1, buffer=1)
    angles, stats = analyze_collision_angles(trajectories)
    
    # Print out overall statistics
    print("\nSummary of Collision Angles:")
    for wall, wall_stats in stats.items():
        print(f"\n{wall.capitalize()} Wall Statistics ({wall_stats['count']} collisions):")
        print(f"  Average Incidence Angle: {wall_stats['incidence_mean']:.2f}° ± {wall_stats['incidence_std']:.2f}°")
        print(f"  Average Reflection Angle: {wall_stats['reflection_mean']:.2f}° ± {wall_stats['reflection_std']:.2f}°")
        print(f"  Mean Angular Error: {wall_stats['mean_error']:.2f}°")