"""kalman.py tests, run against real numpy."""
import numpy as np

from kalman import KalmanFilter


def test_default_shapes_match_dims():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    assert kf.x.shape == (4,)
    assert kf.F.shape == (4, 4)
    assert kf.H.shape == (2, 4)
    assert kf.P.shape == (4, 4)
    assert kf.Q.shape == (4, 4)
    assert kf.R.shape == (2, 2)


def test_predict_advances_state_by_transition_matrix():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([1.0, 2.0])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # position += velocity
    kf.Q = np.zeros((2, 2))

    kf.predict()

    assert np.allclose(kf.x, [3.0, 2.0])


def test_update_moves_state_toward_measurement():
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([0.0])
    kf.H = np.array([[1.0]])
    kf.P = np.array([[1.0]])
    kf.R = np.array([[1.0]])

    kf.update(np.array([10.0]))

    # With equal prior and measurement variance, the update should land halfway.
    assert np.allclose(kf.x, [5.0])


def test_update_shrinks_covariance():
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.H = np.array([[1.0]])
    kf.P = np.array([[1.0]])
    kf.R = np.array([[1.0]])

    kf.update(np.array([1.0]))

    assert kf.P[0, 0] < 1.0
