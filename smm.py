import numpy as np
from sklearn import svm 
from sklearn.model_selection import GridSearchCV

'''
support measure machine

This class implements Support Measure Machine, an 
SVM for distributions usign the kenel mean embedding.
Every SVM method from sklearn.svm can be used.

Parameters
----------
kernel: callabla
    the kernel to be used in the algorithm. 
    Must be a callable. 

base_svm: something like sklearn.svm.SVC

normalize: bool, default=True
           Whether to normalize the induced kernel. 
    
param_grid: None, dict or list of dictionaries,
            if not None, sklearn.model_selection.GridSearchCV
            is used to search parameter values for the
            base SVM.
'''

class smm():

    def __init__(self, kernel, base_svm = svm.SVC, normalize = True,
            param_grid = None, **params):
        self.svm = base_svm(kernel = 'precomputed', **params)
        self.kernel = kernel
        self.normalize = normalize
        self.param_grid = param_grid
        self.training_data = None 
        self.D = None
        self.oneclass = False

    def fit(self, x = None, y = None, verbose = False):
        if not x is None:
            self.compute_gram(x, verbose = verbose) 
        if all([yy == y[0] for yy in y]):
            if verbose:
                print('Number of y values equal to 1')
            self.oneclass = True
            self.thatclass = y[0]
            return self
        if self.param_grid is None:
            self.svm.fit(self.gram, y)
        else:
            grid = GridSearchCV(
                    self.svm,
                    self.param_grid,
                    refit=True
                    )
            grid.fit(self.gram, y) 
            self.svm = grid.best_estimator_

        if verbose:
            print('support vector') 
            print(self.svm.support_)
        return self

    def predict(self, x, isgram = False):
        if self.oneclass:
            return [self.thatclass for i in range(x.shape[0])]   
        if isgram: 
            return self.svm.predict(x)
        else:
            xnew = self.compute_newgram(x, support = self.svm.support_)
            return self.svm.predict(xnew)

    def score(self, x, y, sample_weight = None, isgram = False):
        if self.oneclass:
            pr = self.predict(x, y) 
            return np.average(np.equal(pr, y)) 
        if isgram: 
            return self.svm.score(x, y, sample_weight)
        else:
            xnew = self.compute_newgram(x, support = self.svm.support_)
            return self.svm.score(xnew, y, sample_weight)

    def decision_function(self, x, isgram = False):
        if self.oneclass:
            return [1 for i in range(x.shape[0])]   
        if self.svm.decision_function is None:
            return None
        if isgram: 
            return self.svm.decision_function(x)
        else:
            xnew = self.compute_newgram(x, support = self.svm.support_)
            return self.svm.decision_function(xnew)

    def compute_newgram(self, x, support = None, verbose = False):
        xt = self.training_data
        if support is None:
            support = range(xt.shape[0])
        xnew = np.zeros(shape = [x.shape[0], xt.shape[0]])
        for i in range(x.shape[0]):
            p = np.array(x[i,:].tolist()).transpose()
            if np.size(p.shape) == 1:
                p = p.reshape(-1,1)
            if self.normalize:
                nn =  np.sqrt(self.kernel(p,p).mean())
            for j in support:
                q = np.array(xt[j,:].tolist()).transpose()
                # check shape and reshape in case if 1-dimensional 
                if np.size(q.shape) == 1:
                    q = q.reshape(-1,1)
                #compute scalar product
                xnew[i,j] = self.kernel(p,q).mean() 
                if self.normalize:
                    xnew[i,j] = xnew[i,j] * (self.D[j] / nn ) 
        if verbose:
            print(xnew)
        return(xnew) 
    
    def compute_gram(self, x, verbose = False):
        if x is self.training_data:
            if verbose:
                print('same data not recomputing gram matrix')
            return self 
        else:
            self.training_data = x 
        gram = np.zeros(shape = [x.shape[0], x.shape[0]])
        for i in range(x.shape[0]):
            p = np.array(x[i,:].tolist()).transpose()
            # check shape and reshape in case if 1-dimensional 
            if np.size(p.shape) == 1:
                p = p.reshape(-1,1)
            for j in range(i + 1):
                q = np.array(x[j,:].tolist()).transpose()
                # check shape and reshape in case if 1-dimensional 
                if np.size(q.shape) == 1:
                    q = q.reshape(-1,1)
                # compute scalar product of mean embedding
                gram[i,j] = self.kernel(p,q).mean()
                gram[j,i] = gram[i,j] 
        #save normalizing factors
        self.D = 1 / np.sqrt(gram.diagonal())
        if self.normalize:
            gram = np.diag(self.D) @ gram @ np.diag(self.D)
        #save gram
        self.gram = gram
        if verbose:
            print('Gram matrix:') 
            print(self.gram)
        return self

    def set_data_gram(self, x, gram, D):
        self.training_data = x
        self.gram = gram 
        self.D = D 
        return self 
