import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel 
from sklearn import svm
from smm import smm
from sklearn.metrics import accuracy_score

class meta_causal_smm():

    def __init__(self, base_models, kernel = lambda a,b: rbf_kernel(a, b, 1), 
                 normalize = True, verbose = False,  **kwargs):
        self.verbose = verbose 
        self.kernel = kernel 
        self.normalize = normalize 
        self.base_svm = svm.SVC
        self.kwargs = kwargs 
        self.smm = smm(base_svm = self.base_svm,   
                kernel = self.kernel,
                normalize = self.normalize, **kwargs) 
        if type(base_models) is dict:
            self.base_models = base_models 
        else:
            raise Exception("""base_models argument should be 
                             a dictionary 
                             with base models""")
        

    def fit(self, X, y):
        self.base_models_classifiers = {} 
        #print("start gram computation")
        self.smm.compute_gram(X.to_numpy()) 
        #print("done gram computation")
        # this could run in parallel 
        for nm, cl in  self.base_models.items():
            # obtain base models predictions and scores 
            pred = np.sign(cl.predict(X))
            pred[pred == 0] = -1
            scores = 1 - np.abs(y.to_numpy()[:,0] - pred)
            #pred = cl.predict(X)
            #scores = y.to_numpy()[:,0] * pred
            #print(scores)
            # next we learn smm classifier for each model scores
            self.base_models_classifiers[nm] = smm(base_svm = self.base_svm, 
                    kernel = self.kernel, 
                    normalize = self.normalize, 
                    **self.kwargs).set_data_gram(self.smm.training_data, 
                                                    self.smm.gram, self.smm.D)
            self.base_models_classifiers[nm].fit(None, scores) 
            if self.verbose:
                print(nm)
                print(f"   training score smm: {self.base_models_classifiers[nm].score(self.smm.gram, scores, isgram = True)}") 
                print(f"   average score {nm}: {accuracy_score(y.to_numpy()[:,0], pred)}")

   
    def predict(self, X):
        xnew = self.smm.compute_newgram(X.to_numpy())
        # obtain base models predictions 
        res = np.array([0 for i in range(X.shape[0])])
        for nm, cl in  self.base_models.items():
            pred = np.sign(cl.predict(X))
            #pred = cl.predict(X)
            df = self.base_models_classifiers[nm].decision_function(xnew, isgram = True) 
            #print(nm)
            #print(df)
            res = res + pred * np.exp(df) 
            #res = res + pred * df  #* (1 + np.sign(df)) / 2 
        return(np.sign(res))

    def score(self, X, y):
        xnew = self.smm.compute_newgram(X.to_numpy())
        # obtain base models predictions 
        res = np.array([0 for i in range(X.shape[0])])
        for nm, cl in  self.base_models.items():
            pred = np.sign(cl.predict(X))
            #pred = cl.predict(X)
            df = self.base_models_classifiers[nm].decision_function(xnew, isgram = True) 
            if self.verbose: 
                print(nm)
                scores = 1 - np.abs(y.to_numpy()[:,0] - pred)
                print(f"    score smm: {self.base_models_classifiers[nm].score(xnew, scores, isgram = True)}") 
            #print(nm)
            #print(df)
            res = res + pred * np.exp(df) 
            #res = res + pred * df  #* (1 + np.sign(df)) / 2 
        return(np.average(1 - np.abs(np.sign(res) - y.to_numpy()[:,0])))


