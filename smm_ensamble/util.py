import numpy as np
import jax.numpy as jnp

def cdt_data_to_jax(X):
    tmp = X.to_numpy()
    Xout = np.zeros((tmp.shape[0], tmp[0,0].shape[0], tmp.shape[1]))
    # here we could check dimensions
    for i in range(tmp.shape[0]):
        Xout[i, :, 0] = tmp[i][0] 
        Xout[i, :, 1] = tmp[i][1] 
    return jnp.asarray(Xout)
