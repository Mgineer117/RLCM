import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_and_concatenate_npz(folder):
    states, actions, next_states = [], [], []
    
    for file_path in glob(os.path.join(folder, "*.npz")):
        data = np.load(file_path)
        print(file_path)
        states.append(data['state'])
        actions.append(data['action'])
        next_states.append(data['next_state'])

    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    next_states = np.concatenate(next_states, axis=0)
    return states, actions, next_states

# === Estimation code ===
state, action, next_state = load_and_concatenate_npz("turtlebot_data/")
print(f"Number of sampler: {state.shape[0]}")
delta_t = 0.1

# Estimate derivatives
dx = (next_state[:, 0] - state[:, 0]) / delta_t
dy = (next_state[:, 1] - state[:, 1]) / delta_t
dtheta = (next_state[:, 2] - state[:, 2]) / delta_t

# Prepare regression inputs
theta = state[:, 2]

A_dx = (action[:, 0] * np.cos(theta)).reshape(-1, 1)
A_dy = (action[:, 0] * np.sin(theta)).reshape(-1, 1)
A_dtheta = action[:, 1].reshape(-1, 1)

# Least squares estimation
k1, _, _, _ = np.linalg.lstsq(A_dx, dx, rcond=None)
k2, _, _, _ = np.linalg.lstsq(A_dy, dy, rcond=None)
k3, _, _, _ = np.linalg.lstsq(A_dtheta, dtheta, rcond=None)

print("\nEstimated parameters of B(x):")
print(f"k1 : {k1[0]:.6f}")
print(f"k2 : {k2[0]:.6f}")
print(f"k3 : {k3[0]:.6f}")

# Prediction errors
dx_pred = A_dx @ k1
dy_pred = A_dy @ k2
dtheta_pred = A_dtheta @ k3

dx_error = dx_pred - dx
dy_error = dy_pred - dy
dtheta_error = dtheta_pred - dtheta

mse_dx = np.mean(dx_error ** 2)
mse_dy = np.mean(dy_error ** 2)
mse_dtheta = np.mean(dtheta_error ** 2)

print("\nMean Squared Errors:")
print(f"MSE dx: {mse_dx:.8f}")
print(f"MSE dy: {mse_dy:.8f}")
print(f"MSE dtheta: {mse_dtheta:.8f}")

# === Plot errors ===
time_axis = np.arange(len(dx)) * delta_t

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_axis, dx_error, label='dx error', color='red')
plt.title('dx Prediction Error')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_axis, dy_error, label='dy error', color='green')
plt.title('dy Prediction Error')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_axis, dtheta_error, label='dtheta error', color='blue')
plt.title('dtheta Prediction Error')
plt.legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()