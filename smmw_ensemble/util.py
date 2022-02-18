import numpy as np
from numpy.random import choice
import jax.numpy as jnp

def cdt_data_to_jax(X, size=None):
    tmp = X.to_numpy()
    if size is None:
        size = tmp[0, 0].shape[0]
    Xout = np.zeros((tmp.shape[0], size, tmp.shape[1]))
    # here we could check dimensions
    for i in range(tmp.shape[0]):
        isize = tmp[i][0].shape[0]
        idx = choice(np.arange(isize), size, replace = isize < size)
        Xout[i, :, 0] = tmp[i][0][idx]
        Xout[i, :, 1] = tmp[i][1][idx]
    return jnp.asarray(Xout)
