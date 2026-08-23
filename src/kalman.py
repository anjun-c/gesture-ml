"""
Minimal linear Kalman filter (predict/update only).

Replaces the filterpy dependency, which is unmaintained (last released 2018)
and fails to build on modern setuptools (its setup.py uses the removed
`install_layout` distutils option). detect.py only needs a constant-velocity
predict/update cycle with fixed matrices, so this ~20-line implementation
covers it without an external, install-fragile dependency.
"""
import numpy as np


class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)  # process noise
        self.R = np.eye(dim_z)  # measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x  # residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
