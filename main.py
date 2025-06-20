import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sensor_data.csv")

dt = 1.0  # time step
n = len(data)

# State: [x, y, vx, vy]
x = np.zeros((4, 1))
P = np.eye(4) * 500

# State transition matrix
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Control input matrix
B = np.array([[0.5 * dt**2, 0],
              [0, 0.5 * dt**2],
              [dt, 0],
              [0, dt]])

# Measurement matrix (GPS: pos only)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

R = np.eye(2) * 5      # GPS noise
Q = np.eye(4) * 0.1    # Process noise

est_positions = []

for i in range(n):
    # Control input from IMU
    ax = data.imu_ax[i]
    ay = data.imu_ay[i]
    u = np.array([[ax], [ay]])

    # Predict
    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    # Measurement from GPS
    z = np.array([[data.gps_x[i]],
                  [data.gps_y[i]]])

    # Kalman Gain
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # Update
    y = z - H @ x
    x = x + K @ y
    P = (np.eye(4) - K @ H) @ P

    est_positions.append((x[0,0], x[1,0]))

# Plot
true = data[['gps_x', 'gps_y']].values
est = np.array(est_positions)

plt.plot(true[:,0], true[:,1], label="GPS (Noisy)")
plt.plot(est[:,0], est[:,1], label="EKF Estimate")
plt.legend()
plt.title("Vehicle Position Estimation - EKF Fusion")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()
