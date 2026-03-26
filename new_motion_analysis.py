import numpy as np
from scipy.optimize import minimize
from puck_data_2 import segment_puck_trajectory
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
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
    # return - np.arctan(v_norm ** np.sqrt(B / f)) / np.sqrt(B * f)

    return -np.arctan(v_norm * np.sqrt(B / f)) / np.sqrt(B * f)


def getD(B, mass, f, C):
    """
    Returns constant D to satisfy the initial condition.
    """
    return -(mass / B) * np.log(np.cos(np.sqrt(B * f) * C))


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
    initial_guess = [0.037, 1e-5, 1e-4] #actual mass is 37g (full size puck with jacket)

    # Set bounds for the parameters:
    # mass: between 1e-6 and 0.1 kg, f: between 1e-8 and 1e-2, B: between 1e-8 and 1e-1.
    bounds = [(1e-6, 0.1), (1e-8, 1e-2), (1e-8, 1e-1)]

    result = minimize(
        objective, initial_guess, args=(segments,), bounds=bounds, method="L-BFGS-B"
    )

    return result


# Example usage:
if __name__ == "__main__":
    # Replace this with your actual data (n, 3) numpy array.
    # First row: initial position (x, y) and dt (can be set to 0)
    # Subsequent rows: position (x, y) and dt (time since previous measurement)

    PROJECT_PATH = Path(__file__).resolve().parents[0]
    datacsv = f"{PROJECT_PATH}/data/puck_movement_data.csv"
    datacsv2 = f"{PROJECT_PATH}/data/puck_movement_data2.csv"
    # datacsv3 = f"{PROJECT_PATH}/data/mar-12-normal-2.csv"
    # datacsv4 = f"{PROJECT_PATH}/data/mar-12-large-angle.csv"
    # datacsv5 = f"{PROJECT_PATH}/data/position-mar-14-1.csv"

    segments, _ = segment_puck_trajectory(datacsv)
    segments2, _ = segment_puck_trajectory(datacsv2)
    # segments3, _ = segment_puck_trajectory(datacsv3)
    # segments4, _ = segment_puck_trajectory(datacsv4)
    # segments5, _ = segment_puck_trajectory(datacsv5)
    # all_segments = segments + segments2 + segments3 + segments4 + segments5
    all_segments = segments + segments2
    print(len(all_segments))
    min_length = 11
    results = []
    result = optimize_parameters(all_segments)
    mass, f, B = result.x
    if result.success:
        print("Optimized parameters:")
        print("mass =", mass, "kg")
        print("friction f =", f)
        print("damping B =", B)
    else:
        print("Optimization did not converge. Message:", result.message)
        print("Using best parameters found so far for plotting.")

    # Plot the optimized model against measured data with an interactive segment slider.
    plottable_segments = []
    for seg in all_segments:
        if len(seg) > 10:
            dt_check = seg[1:, 2]
            if np.sum(dt_check[:9]) > 0:
                plottable_segments.append(seg)

    if not plottable_segments:
        raise ValueError(
            "No plottable segments found (need len > 10 and positive sum(dt[:9]))."
        )

    def compute_series(segment_data):
        dt_local = segment_data[1:, 2]
        times_local = np.cumsum(dt_local)
        initial_position_local = segment_data[0, :2]
        measured_local = np.sqrt(
            np.sum((segment_data[1:, :2] - initial_position_local) ** 2, axis=1)
        )
        v_norm_local = measured_local[9] / np.sum(dt_local[:9])
        predicted_local = model_displacement([mass, f, B], times_local, v_norm_local)
        return times_local, measured_local, predicted_local

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    (line_measured,) = ax.plot([], [], label="Measured displacement")
    (line_predicted,) = ax.plot([], [], label="Predicted displacement")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.legend()

    slider_ax = fig.add_axes((0.18, 0.08, 0.64, 0.04))
    segment_slider = Slider(
        ax=slider_ax,
        label="Segment",
        valmin=0,
        valmax=len(plottable_segments) - 1,
        valinit=0,
        valstep=1,
    )

    def update_plot(_):
        seg_idx = int(segment_slider.val)
        times_plot, measured_plot, predicted_plot = compute_series(
            plottable_segments[seg_idx]
        )

        line_measured.set_data(times_plot, measured_plot)
        line_predicted.set_data(times_plot, predicted_plot)
        ax.relim()
        ax.autoscale_view()
        ax.set_title(f"Model Fit to Segment {seg_idx + 1}/{len(plottable_segments)}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        current = int(segment_slider.val)
        if event.button == "up":
            next_idx = min(current + 1, len(plottable_segments) - 1)
        elif event.button == "down":
            next_idx = max(current - 1, 0)
        else:
            return
        segment_slider.set_val(next_idx)

    segment_slider.on_changed(update_plot)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    update_plot(None)
    plt.show()
