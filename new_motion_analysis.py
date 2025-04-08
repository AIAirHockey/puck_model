import numpy as np
from scipy.optimize import minimize
from puck_data_2 import segment_puck_trajectory
from matplotlib import pyplot as plt
from pathlib import Path

def get_tPZero(mass, C):
    return -C * mass

def dP(t, B, f, mass, C, D):
    """
    Returns the predicted total displacement p at time t.
    t is capped by tpZero = -C*mass.
    """
    tpZero = get_tPZero(mass, C)
    # Limit t to not exceed the valid range
    t_adj = np.minimum(tpZero, t)
    p = (mass / B) * np.log(np.cos((np.sqrt(B * f) * (C * mass + t_adj)) / mass)) + D
    return p

def getC(v_norm, B, f):
    """
    Returns constant C computed from the fixed initial velocity, damping B, and friction f.
    """
    return - np.arctan(v_norm ** np.sqrt(B / f)) / np.sqrt(B * f)

def getD(B, mass, f, C):
    """
    Returns constant D to satisfy the initial condition.
    """
    return - (mass / B) * np.log(np.cos(np.sqrt(B * f) * C))

def model_displacement(params, times, v_norm):
    """
    Given the parameters [mass, f, B] and cumulative times, compute predicted displacement.
    v_norm is fixed (computed from the initial data point).
    """
    mass, f, B = params
    C = getC(v_norm, B, f)
    D = getD(B, mass, f, C)
    return dP(times, B, f, mass, C, D)

def objective(params, segments):
    """
    Mean squared error between model predictions and measured displacements.
    """
    errors = []
    for data in segments:
        if len(data) > 10:
                
            dt = data[1:, 2]
            # Compute cumulative times from dt values
            times = np.cumsum(dt)
            
            # Compute measured displacement: Euclidean distance from the initial position
            initial_position = data[0, :2]
            measured = np.sqrt(np.sum((data[1:, :2] - initial_position) ** 2, axis=1))
            
            # Compute the fixed initial velocity from the first five displacements and dts.
            
            v_norm = measured[9] / np.sum(dt[:9])

            predicted = model_displacement(params, times, v_norm)
            mse = np.mean((predicted - measured) ** 2)
            errors.append(mse)

    MSE = np.mean(np.array(errors) ** 2)
    return MSE

def optimize_parameters(segments):
    """
    data: numpy array of shape (n, 3) where each row is (x, y, dt).
          dt is undefined (or can be set to 0) for the first row.
    
    Returns:
      result: optimization result containing the optimized parameters (mass, f, B)
      v_norm: the computed initial velocity
    """
    
    # Initial guess for the parameters: mass (in kg), friction, and damping.
    initial_guess = [0.008, 1e-5, 1e-4]
    
    # Set bounds for the parameters:
    # mass: between 1e-6 and 0.1 kg, f: between 1e-8 and 1e-2, B: between 1e-8 and 1e-1.
    bounds = [(1e-6, 0.1), (1e-8, 1e-2), (1e-8, 1e-1)]
    
    result = minimize(objective, initial_guess, args=(segments),
                      bounds=bounds, method='L-BFGS-B')
    

    return result

# Example usage:
if __name__ == "__main__":
    # Replace this with your actual data (n, 3) numpy array.
    # First row: initial position (x, y) and dt (can be set to 0)
    # Subsequent rows: position (x, y) and dt (time since previous measurement)

    PROJECT_PATH = Path(__file__).resolve().parents[0]
    datacsv = f"{PROJECT_PATH}/data/position-mar-11-2.csv"
    datacsv2 = f"{PROJECT_PATH}/data/mar-12-normal-1.csv"
    datacsv3 = f"{PROJECT_PATH}/data/mar-12-normal-2.csv"
    datacsv4 = f"{PROJECT_PATH}/data/mar-12-large-angle.csv"
    datacsv5 = f"{PROJECT_PATH}/data/position-mar-14-1.csv"

    segments, _ = segment_puck_trajectory(datacsv)
    segments2, _ = segment_puck_trajectory(datacsv2)
    segments3, _ = segment_puck_trajectory(datacsv3)
    segments4, _ = segment_puck_trajectory(datacsv4)
    segments5, _ = segment_puck_trajectory(datacsv5)
    all_segments = segments + segments2 + segments3 + segments4 + segments5
    min_length = 11
    results = []
    result = optimize_parameters(all_segments)
    if result.success:
        mass, f, B = result.x
        print("Optimized parameters:")
        print("mass =", mass, "kg")
        print("friction f =", f)
        print("damping B =", B)
    else:
        print("Optimization did not converge. Message:", result.message)

    # Plot the optimized model against the measured data
    segment = segments[7]
    times = np.cumsum([point[2] for point in segment[1:]])
    measured = np.sqrt(np.sum((np.array([point[:2] for point in segment[1:]]) - segment[0][:2]) ** 2, axis=1))
    #result, v_norms = optimize_parameters(segment)
    data = segment
    dt = data[1:, 2]
    # Compute cumulative times from dt values
    
    # Compute measured displacement: Euclidean distance from the initial position
    initial_position = data[0, :2]
    measured = np.sqrt(np.sum((data[1:, :2] - initial_position) ** 2, axis=1))
    
    # Compute the fixed initial velocity from the first five displacements and dts.
    v_norm = measured[9] / np.sum(dt[:9])
    # B_avg = 0.001
    C = getC(v_norm, B, f)
    D = getD(B, mass, f, C)
    predicted = model_displacement([mass, f, B], times, v_norm)
    plt.plot(times, measured, label='Measured displacement')
    plt.plot(times, predicted, label='Predicted displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Model Fit to Measured Data')
    plt.legend()
    plt.show()
    

