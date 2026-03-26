import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load CSV
df = pd.read_csv(r"C:\Users\Ian\Documents\AirHockey\repos\puck_model\data\feedforward_feedback_data_delayed (12).csv")

# Remove invalid dt rows (dt <= 0)
df = df[df["dt"] > 0].reset_index(drop=True)

# Convert dt from milliseconds to seconds
dt_ms = df["dt"].to_numpy()
dt = dt_ms / 1000.0     # <-- IMPORTANT

# Position data
x = df["x"].to_numpy()
y = df["y"].to_numpy()

# ---------------------------------------------
# Smoothing parameters tuned for 10,000 samples
# ---------------------------------------------
win = 51   # ~0.1 seconds, good for 1 kHz data
poly = 3

# -----------------------------
# 1. Smooth position
# -----------------------------
x_s = savgol_filter(x, win, poly)
y_s = savgol_filter(y, win, poly)

# -----------------------------
# 2. Velocity (finite difference)
# -----------------------------
vx = np.diff(x_s) / dt[:-1]
vy = np.diff(y_s) / dt[:-1]

# Pad to original length
vx = np.concatenate(([vx[0]], vx))
vy = np.concatenate(([vy[0]], vy))

# Smooth velocity
vx_s = savgol_filter(vx, win, poly)
vy_s = savgol_filter(vy, win, poly)

# -----------------------------
# 3. Acceleration
# -----------------------------
ax = np.diff(vx_s) / dt[:-1]
ay = np.diff(vy_s) / dt[:-1]

ax = np.concatenate(([ax[0]], ax))
ay = np.concatenate(([ay[0]], ay))

# Smooth acceleration
ax_s = savgol_filter(ax, win, poly)
ay_s = savgol_filter(ay, win, poly)

# -----------------------------
# 4. Acceleration magnitude
# -----------------------------
acc_mag = np.sqrt(ax_s**2 + ay_s**2)

# Time vector
time = np.cumsum(dt)

# -----------------------------
# 5. Maximum acceleration
# -----------------------------
max_acc = np.max(acc_mag)
print("Maximum acceleration magnitude:", max_acc)

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(time, acc_mag)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration Magnitude (Smoothed) (m/s)")
plt.title("Smoothed Acceleration Magnitude Over Time")
plt.grid(True)
plt.show()
