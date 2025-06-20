import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv("vehicle_sensor_fusion_dataset.csv")
ekf_x, ekf_y = [], []

# Video setup
video_name = 'ekf_path_simulation.avi'
frame_size = (640, 480)
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 5, frame_size)

for i in range(len(df)):
    plt.figure(figsize=(6.4, 4.8))
    # Plot GPS
    plt.scatter(df.gps_x[:i], df.gps_y[:i], color='blue', label="GPS", alpha=0.5, s=10)
    # Plot EKF path (you can use actual saved EKF outputs here)
    if i > 0:
        ekf_x.append(df.gps_x[i]*0.8 + df.gps_x[i-1]*0.2)  # dummy fusion for visual
        ekf_y.append(df.gps_y[i]*0.8 + df.gps_y[i-1]*0.2)
        plt.plot(ekf_x, ekf_y, color='red', label="EKF Estimate", linewidth=2)

    plt.title(f"Frame {i}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save frame as image
    plt.savefig("frame.jpg")
    plt.close()

    # Read frame as image and write to video
    frame = cv2.imread("frame.jpg")
    resized_frame = cv2.resize(frame, frame_size)
    out.write(resized_frame)

out.release()
print("✅ Video saved as:", video_name)
