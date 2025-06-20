import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("vehicle_sensor_fusion_dataset.csv")
dt = 1.0
n = len(df)

# Initialize state vector: [x, y, vx, vy, heading, yaw_rate]
x = np.zeros((6, 1))
P = np.eye(6) * 500

# Matrices
F = np.eye(6)
F[0, 2] = dt  # x += vx*dt
F[1, 3] = dt  # y += vy*dt
F[4, 5] = dt  # heading += yaw_rate*dt

B = np.zeros((6, 2))
B[2, 0] = dt   # vx += ax*dt
B[3, 1] = dt   # vy += ay*dt

H = np.zeros((2, 6))
H[0, 0] = 1  # gps_x measures x
H[1, 1] = 1  # gps_y measures y

Q = np.eye(6) * 0.1
R = np.eye(2) * 4.0

est_positions = []

for i in range(n):
    ax = df.imu_ax[i]
    ay = df.imu_ay[i]
    u = np.array([[ax], [ay]])

    # Prediction
    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    # Measurement (GPS)
    z = np.array([[df.gps_x[i]], [df.gps_y[i]]])

    # Kalman Gain
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # Update
    y = z - H @ x
    x = x + K @ y
    P = (np.eye(6) - K @ H) @ P

    est_positions.append((x[0, 0], x[1, 0]))

# Plot
gps = df[['gps_x', 'gps_y']].values
est = np.array(est_positions)

plt.figure(figsize=(10, 6))
plt.plot(gps[:, 0], gps[:, 1], label="Noisy GPS", alpha=0.6)
plt.plot(est[:, 0], est[:, 1], label="EKF Estimate", linewidth=2)
plt.title("Extended Kalman Filter (GPS + IMU + Wheel Encoder)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt_path = "vehicle_sensor_fusion_dataset.csv.png"
plt.savefig(plt_path)
plt_path
