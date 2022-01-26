from sklearn import svm
from sklearn.metrics import accuracy_score
from typing import Optional, Callable, Dict
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from utilities import cdt_data_to_numpy
from sklearn.model_selection import GridSearchCV


def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum((x - y) ** 2)


def rbf_k(x: jnp.ndarray, y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Radial Basis Function (RBF).
    The most popular kernel in all of kernel methods.
    .. math::
        k(\mathbf{x,y}) = \\
                \\exp \left( - \\gamma\\
                ||\\mathbf{x} - \\mathbf{y}||^2_2\\
                \\right) 
    Parameters
    ----------
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)
    gamma: float
        the gamma parameter of the rbf_kernel 
    Returns
    -------
    kernel_mat : jax.numpy.ndarray (n_samples, n_samples)
        the kernel matrix 
    References
    ----------
    .. [1] David Duvenaud, *Kernel Cookbook*
    """
    rbf = lambda xx,yy: jnp.exp(-gamma * jnp.sum((xx - yy) ** 2))
    return vmap(lambda x1: vmap(lambda y1: rbf(x1, y1))(y))(x)


@jit
def kme_rbf_k(p: jnp.ndarray, q: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Kernel Mean Embedding 
    This function computes the kernel matrix (or Gram matrix)
    between the kernel mean embeddings of the samples in p and q. 

    Parameters
    ----------
    p : jax.numpy.ndarray
        input dataset (n_samples, n_samplesize, n_features)
    q : jax.numpy.ndarray
        other input dataset (n_samples, n_sampelsize, n_features)
    kernel: callable
        the base kernel function (e.g. rbf_kernel)  
    Returns
    -------
    kernel_mat : jax.numpy.ndarray (n_samples, n_samples)
        the kernel matrix 
    """
    loop =  lambda p1: jax.lax.map(lambda q1: jnp.mean(rbf_k(p1, q1, gamma)), p)
    return jax.lax.map(loop, q) 


class SMMEnsamble():


    def __init__(self, base_models,
            gamma = 1.0, 
            exp_weights = False,
            param_grid = None, 
            verbose = False):
        if type(base_models) is dict:
            self.base_models = base_models 
        else:
            raise Exception("""base_models argument should be 
                             a dictionary 
                             with base models""")
        self.verbose = verbose 
        self.gamma = gamma 
        self.exp_weights = exp_weights
        self.param_grid = param_grid


    def fit(self, X, y):
        Xn = cdt_data_to_numpy(X)
        self.base_models_classifiers = {} 
        # save trainign data
        self.X = Xn
        # compute gram matrix 
        gram = np.asarray(kme_rbf_k(Xn, Xn, self.gamma))
        # this could run in parallel 
        for nm, cl in  self.base_models.items():
            # obtain base models predictions and scores 
            pred = np.sign(cl.predict(X))
            # score is +1 if base model is correct, -1 otherwise
            scores = 1 - np.abs(y.to_numpy()[:,0] - pred)
            scores[scores == 0] = -1
            # next we learn smm classifier for each model scores
            model =  svm.SVC(kernel = "precomputed")
            if self.param_grid is None:
                model.fit(gram, scores)
            else:
                grid = GridSearchCV(
                        model,
                        self.param_grid,
                        refit=True
                        )
                grid.fit(gram, scores) 
                model = grid.best_estimator_
            # save classifier for the method
            self.base_models_classifiers[nm] = model 
            if self.verbose:
                print(nm)
                print(f"training score smm: {model.score(gram, scores)}") 
                print(f"accuracy score {nm}: {accuracy_score(y.to_numpy()[:,0], pred)}")
        return self


    def predict(self, X):
        Xn = cdt_data_to_numpy(X)
        # compute test gram 
        xnew = np.asarray(kme_rbf_k(self.X, Xn, self.gamma))
        
        res = np.array([0 for i in range(Xn.shape[0])])
        for nm, cl in  self.base_models.items():
            # obtain base model predictions 
            pred = np.sign(cl.predict(X))
            # get classifier decision function
            model = self.base_models_classifiers[nm]
            df = model.decision_function(xnew) 
            if self.exp_weights:
                df = np.exp(df)
            res = res + pred * df 
        return(np.sign(res))


    def score(self, X, y):
        pred = self.predict(X) 
        return accuracy_score(y, pred)
