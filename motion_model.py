import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from puck_data_2 import segment_puck_trajectory, compute_v0, get_time_array


PROJECT_PATH = str(Path(__file__).resolve().parents[0])

def ode_model(t, y, a, b):
    """
    Defines the ODE system:
      y[0] = x (displacement)
      y[1] = x_dot (velocity)
    
    The system is:
      dx/dt = x_dot
      d(x_dot)/dt = -a - b*(x_dot)**2
    """
    return [y[1], -a - b*(y[1]**2)]

def integrate_model(t, a, b, v0):
    """
    Solves the ODE from t[0] to t[-1] with initial conditions x(0)=0, x'(0)=v0.
    
    Parameters:
      t  : array of time points
      a  : coefficient from the ODE
      b  : coefficient from the ODE
      v0 : initial velocity
      
    Returns:
      x : displacement values computed at the times in t.
    """
    sol = solve_ivp(ode_model, (t[0], t[-1]), [0, v0], args=(a, b), t_eval=t, method='RK45', max_step=0.1, rtol=1e-2, atol=1e-5)

    if sol.status != 0:
        print(f"Warning: solve_ivp did not successfully integrate over the entire range. Status: {sol.status}")
    print(f"integrate_model: t length = {len(t)}, sol.y[0] length = {len(sol.y[0])}")
    return sol.y[0]

def fit_trajectory(time, positions, v0):
    """
    Fits the ODE model to experimental trajectory data.
    
    Parameters:
      time      : array of time points corresponding to the data.
      positions : Nx2 array (or list) of [x, y] positions.
      v0        : initial velocity (given separately).
      
    Process:
      - Shifts the experimental positions so that the start is at (0,0).
      - Computes the displacement (Euclidean distance from start) at each time.
      - Uses curve_fit to determine best-fit parameters a and b by comparing
        the integrated model solution to the experimental displacement.
        
    Returns:
      fitted_a       : best-fit parameter a
      fitted_b       : best-fit parameter b
      displacement   : experimental displacement (shifted so that x(0)=0)
      fitted_traj    : displacement computed from the fitted model.
    """
    positions = np.array(positions)
    # Shift positions so that the initial position is at the origin.
    shifted = positions - positions[0]
    # Compute the displacement (magnitude)
    displacement = np.sqrt(shifted[:,0]**2 + shifted[:,1]**2)
    
    # Debugging prints
    print(f"time length: {len(time)}")
    print(f"displacement length: {len(displacement)}")
    
    def model_for_fit(t, a, b):
        # Integrate over the full time array
        full_model = integrate_model(time, a, b, v0)
        # Debugging prints
        print(f"full_model length: {len(full_model)}")
        print(f"full_model: {full_model}")
        # Interpolate the integrated model at the t values provided
        interpolated_model = np.interp(t, time, full_model)
        print(f"interpolated_model length: {len(interpolated_model)}")
        return interpolated_model
    
    # Initial guess for a and b
    p0 = [0, 0]
    
    # Now, curve_fit will call model_for_fit with various t arrays,
    # and the interpolation ensures the output always matches t's shape.
    popt, _ = curve_fit(model_for_fit, time, displacement, p0=p0)
    fitted_a, fitted_b = popt
    
    # Get the full fitted trajectory using the best-fit parameters.
    fitted_traj = integrate_model(time, fitted_a, fitted_b, v0)
    
    return fitted_a, fitted_b, displacement, fitted_traj

def generate_noisy_trajectory(a, b, v0, t, noise_mean=0.0, noise_std=0.2):
    """
    Generates a trajectory from the ODE x'' = -a - b*(x')^2 with initial conditions
    x(0)=0 and x'(0)=v0, and then adds normally distributed noise to each point.
    
    Parameters:
      a         : coefficient from the ODE.
      b         : coefficient from the ODE.
      v0        : initial velocity.
      t         : array of time points at which to compute the solution.
      noise_mean: mean of the normal noise.
      noise_std : standard deviation of the normal noise.
      
    Returns:
      noisy_x   : the trajectory with added noise.
    """
    true_traj = integrate_model(t, a, b, v0)
    noise = np.random.normal(noise_mean, noise_std, size=true_traj.shape)
    return true_traj + noise

if __name__ == "__main__":
    # -----------------------
    # Example demonstration:
    # -----------------------
    
    # Define true parameters for synthetic data generation.
    true_a = 0.001
    true_b = 3
    true_v0 = 3.0  # initial velocity
    
    # Generate a set of time points (for example, 10 seconds sampled 100 times)
    t = np.linspace(0, 1, 100)
    
    # Generate a synthetic trajectory with noise using the true parameters.
    # Here we only simulate motion in one direction (y remains 0), so the positions
    # are taken as (noisy displacement, 0).
    noisy_traj = generate_noisy_trajectory(true_a, true_b, true_v0, t, noise_mean=0, noise_std=0.03)
    positions = np.column_stack((noisy_traj, np.zeros_like(noisy_traj)))

    data_csv = PROJECT_PATH + '/data/position-mar-11-2.csv'
    puck_data = pd.read_csv(data_csv)
    segment_number = 12
    segmented_positions, _ = segment_puck_trajectory(data_csv, accel_threshold=0.5, buffer=1)
    v01 = compute_v0(segmented_positions[segment_number])
    t1 = get_time_array(segmented_positions[segment_number])

    segment = np.array(segmented_positions[segment_number])
    xypositions = np.column_stack((segment[:, 0], segment[:, 1]))

    a, b, experimental_disp, fitted_traj = fit_trajectory(t1, xypositions, v01)
    print(f"Fitted parameters: a = {a}, b = {b}")

    
    # # Fit the model to the synthetic data.
    # fitted_a, fitted_b, experimental_disp, fitted_traj = fit_trajectory(t, positions, true_v0)
    
    # print("Fitted parameters:")
    # print("a =", fitted_a)
    # print("b =", fitted_b)
    
    # # Plot the experimental (shifted) displacement and the fitted model trajectory.
    # plt.figure(figsize=(10, 6))
    # plt.plot(t, experimental_disp, 'o', label="Experimental (shifted) displacement")
    # plt.plot(t, fitted_traj, '-', label="Fitted model")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Displacement")
    # plt.title("Air Hockey Puck Trajectory Fit")
    # plt.legend()
    # plt.show()
    #fit the model to the experimental data
    plt.figure(figsize=(10, 6))
    plt.plot(t1, experimental_disp, 'o', label="Experimental (shifted) displacement")
    plt.plot(t1, fitted_traj, '-', label="Fitted model")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.title("Air Hockey Puck Trajectory Fit")
    plt.legend()
    plt.show()
