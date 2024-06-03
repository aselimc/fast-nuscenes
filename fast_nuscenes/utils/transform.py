from numpy.typing import ArrayLike

import numpy as np


def apply_transformation(transformation: ArrayLike, input_data: ArrayLike) -> ArrayLike:
    """
    Apply a transformation to a 2D or 3D array of points.
    
    Args:
    -transformation: 2D array of transformation matrix
    -input_data: 2D or 3D array of points
    
    Returns:
    -x_t: 2D or 3D array of transformed points
    """
    
    assert len(input_data.shape) == 2 or len(input_data.shape) == 3, "data should be a 2D or 3D array"
    assert len(transformation.shape) == 2, "transformation should be a 2D array"
    assert transformation.shape[0] == transformation.shape[1], "transformation should be square matrix"

    if len(input_data.shape) == 2:
        d = input_data.shape[1]
        input_data_ = input_data
    else:
        d = input_data.shape[2]
        input_data_ = input_data.reshape(-1, 3)

    assert transformation.shape[0] == d + 1, "transformation dimension mismatch"
    assert np.max(np.abs(transformation[-1, :] - np.r_[np.zeros(d), 1])) <= np.finfo(float).eps * 1e3, "bad transformation"

    x_t = np.c_[input_data_, np.ones((input_data_.shape[0], 1))] @ transformation.T
    x_t = x_t[:, :d].reshape(input_data.shape)

    return x_t
