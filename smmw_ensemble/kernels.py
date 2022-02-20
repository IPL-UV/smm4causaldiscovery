import jax
import jax.numpy as jnp



def rbf_2d(x: jnp.ndarray, y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Radial Basis Function (RBF) for 2d.

    Parameters
    ----------
    x : jax.numpy.ndarray
        input dataset (sample_size_x, n_features)
    y : jax.numpy.ndarray
        other input dataset (sample_size_y, n_features)
    gamma: float
        the gamma parameter of the rbf_kernel 
        if gamma<0 the median distance is used
    Returns
    -------
    kernel_mat : jax.numpy.ndarray (sample_size_x, sample_size_y)
        the kernel matrix 
    """
    dist = jnp.power(x[:,0,None] - y[:,0], 2) +  jnp.power(x[:,1,None] - y[:,1], 2)
    return jnp.exp( - gamma * dist)


@jax.jit
def kme_rbf(p: jnp.ndarray, q: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Kernel Mean Embedding 
    This function computes the kernel matrix (or Gram matrix)
    between the kernel mean embeddings of the samples in p and q. 
    The base kernel used is the 2d RBF kernel. 
    Parameters
    ----------
    p : jax.numpy.ndarray
        input dataset (n_samples_p, n_samplesize, n_features)
    q : jax.numpy.ndarray
        other input dataset (n_samples_q, n_sampelsize, n_features)
    gamma: float 
    Returns
    -------
    kernel_mat : jax.numpy.ndarray (n_samples_p, n_samples_q)
        the kernel matrix between the KMEs 
    """
    loop =  lambda p1: jax.lax.map(lambda q1: jnp.mean(rbf_2d(p1, q1, gamma)), q)
    return jax.lax.map(loop, p) 

def rbf_2d_median(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Radial Basis Function (RBF) for 2d.

    Parameters
    ----------
    x : jax.numpy.ndarray
        input dataset (sample_size_x, n_features)
    y : jax.numpy.ndarray
        other input dataset (sample_size_y, n_features)
    gamma: float
        the gamma parameter of the rbf_kernel 
        if gamma<0 the median distance is used
    Returns
    -------
    kernel_mat : jax.numpy.ndarray (sample_size_x, sample_size_y)
        the kernel matrix 
    """
    dist = jnp.power(x[:,0,None] - y[:,0], 2) +  jnp.power(x[:,1,None] - y[:,1], 2)
    gamma = 1/jnp.median(dist)
    return jnp.exp( - gamma * dist)


@jax.jit
def kme_rbf_median(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Kernel Mean Embedding 
    This function computes the kernel matrix (or Gram matrix)
    between the kernel mean embeddings of the samples in p and q. 
    The base kernel used is the 2d RBF kernel. 
    Parameters
    ----------
    p : jax.numpy.ndarray
        input dataset (n_samples_p, n_samplesize, n_features)
    q : jax.numpy.ndarray
        other input dataset (n_samples_q, n_sampelsize, n_features)
    gamma: float 
    Returns
    -------
    kernel_mat : jax.numpy.ndarray (n_samples_p, n_samples_q)
        the kernel matrix between the KMEs 
    """
    loop =  lambda p1: jax.lax.map(lambda q1: jnp.mean(rbf_2d_median(p1, q1)), q)
    return jax.lax.map(loop, p) 
