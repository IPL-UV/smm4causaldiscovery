from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import time
from .util import cdt_data_to_jax
from .kernels import kme_rbf 
from os import cpu_count
from joblib import Parallel, delayed

class SMMwEnsemble():

    def __init__(self, base_methods,
            include_constant=True,
            gamma=1.0, 
            exp_weights=False,
            param_grid=None, 
            C=1.0,
            parallel=True,
            njobs=cpu_count() - 1,
            verbose=False):
        if type(base_methods) is dict:
            self.base_methods = base_methods 
        else:
            raise Exception("""base_methods argument should be 
                             a dictionary 
                             with base methods""")

        self.include_constant = include_constant
        self.verbose = verbose 
        self.gamma = gamma 
        self.C = C
        self.exp_weights = exp_weights
        self.param_grid = param_grid
        self.parallel = parallel
        self.njobs = njobs

    def fit(self, X, y):
        if self.include_constant:
            self.base_methods.update({"_constant": None})
        # data to jax array
        jX = cdt_data_to_jax(X)
        self.base_methods_classifiers = {} 
        self.base_scores = {} 
        self.smm_scores = {} 
        self.oneclass_signs = {}

        # save training data
        self.Xtrain = jX

        # compute gram matrix 
        start = time.time()
        gram = kme_rbf(jX, jX, self.gamma)
        end = time.time()

        if self.verbose:
            print(f'gram computation in {end - start} seconds')

       # get all the base methods predictions 
        if self.parallel:
            base_preds = Parallel(n_jobs=self.njobs)\
                    (delayed(_predict_base)(nm, cl, X) \
                                for nm, cl in self.base_methods.items())
        else:
            base_preds = (_predict_base(nm, cl, X) for nm, cl in self.base_methods.items())

        # train base classifiers 
        for nmpred in base_preds:
            nm = nmpred[0]
            pred = np.sign(nmpred[1])

            scores = 1 - np.abs(y.to_numpy()[:,0] - pred)
            scores[scores == 0] = -1
            if all([score == scores[0] for score in scores]):
                if self.verbose:
                    print(f'score for {nm} is always {scores[0]}')
        
                model = svm.OneClassSVM(kernel='precomputed')
                one_class = True
                oneclass_sign = scores[0]
            else:
                model =  svm.SVC(kernel="precomputed", C=self.C)
                one_class = False
                oneclass_sign = 1.0
            if self.param_grid is None or one_class:
                model.fit(gram, scores)
            else:
                grid = GridSearchCV(
                        model,
                        self.param_grid,
                        refit=True)
                grid.fit(gram, scores) 
                model = grid.best_estimator_
            # save classifier for the method
            self.base_methods_classifiers[nm] = model
            self.oneclass_signs[nm] = oneclass_sign
            if one_class:
                self.smm_scores[nm] = 1.0
            else:
                self.smm_scores[nm] = model.score(gram, scores)
            self.base_scores[nm] = accuracy_score(y.to_numpy()[:,0], pred) 
            if self.verbose:
                print(nm)
                print(f"training score smm: {self.smm_scores[nm]}") 
                print(f"accuracy score {nm}: {self.base_scores[nm]}")

        self.best_base = max(self.base_scores, key=self.base_scores.get)
        return self



    def predict(self, X):
        jX = cdt_data_to_jax(X)
        # compute test gram 
        start = time.time()
        xnew = kme_rbf(jX, self.Xtrain, self.gamma)
        end = time.time()
        if self.verbose:
            print(f'test gram computed in {end - start} seconds')

        # get all the base methods predictions 
        self.base_predictions = {}
        if self.parallel:
            base_preds = Parallel(n_jobs=self.njobs)\
                    (delayed(_predict_base)(
                        nm,
                        cl, 
                        X) \
                                for nm, cl in self.base_methods.items())
        else:
            base_preds = (_predict_base(nm, cl, X) for nm, cl in self.base_methods.items())

        for pred in base_preds:
            self.base_predictions[pred[0]] = pred[1] 

        res = np.zeros(jX.shape[0])
        # compute weighted average, simple average, voting
        for nm, pr in self.base_predictions.items(): 
            model = self.base_methods_classifiers[nm]
            # get classifier decision function
            df = model.decision_function(xnew) * self.oneclass_signs[nm] 
            if self.exp_weights:
                df = np.exp(df)
            res += np.sign(pr) * df 

        return np.sign(res) 

    def score_base(self, y):
        if self.base_predictions is None:
            return None
        scores = {}
        for nm, pr in self.base_predictions.items():
            scores[nm] = accuracy_score(np.sign(pr), y)
        return scores


    def score_alternatives(self, y):
        if self.base_predictions is None:
            return None
        avg = np.zeros(y.shape[0])
        vot = np.zeros(y.shape[0])
        best = np.zeros(y.shape[0])
        # compute weighted average, simple average, voting
        for nm, pr in self.base_predictions.items(): 
            if nm != "_constant":
               print(nm)
               avg += pr 
               vot += np.sign(pr)
            if nm == self.best_base:
                best = np.sign(pr)
        scores = { "avg": accuracy_score(np.sign(avg), y), 
                   "vot": accuracy_score(np.sign(vot), y),
                   "best": accuracy_score(np.sign(best), y)}
        return scores


    def score(self, X, y):
        pred = self.predict(X) 
        return accuracy_score(y, pred)


def _predict_base(nm, cl, X):
    # obtain base model predictions 
    if cl is None:
        return nm, np.ones(X.shape[0])
    else:
        return nm, cl.predict(X)
