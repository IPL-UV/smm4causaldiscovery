from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import time
from .util import cdt_data_to_jax
from .kernels import kme_rbf 

class SMMEnsamble():

    def __init__(self, base_models,
            gamma=1.0, 
            exp_weights=False,
            param_grid=None, 
            C=1.0,
            verbose = False):
        if type(base_models) is dict:
            self.base_models = base_models 
        else:
            raise Exception("""base_models argument should be 
                             a dictionary 
                             with base models""")
        self.verbose = verbose 
        self.gamma = gamma 
        self.C = C
        self.exp_weights = exp_weights
        self.param_grid = param_grid

    def fit(self, X, y):
        # data to jax array
        jX = cdt_data_to_jax(X)
        self.base_models_classifiers = {} 
        self.base_scores = {} 
        self.smm_scores = {} 
        # save training data
        self.Xtrain = jX

        # compute gram matrix 
        start = time.time()
        gram = kme_rbf(jX, jX, self.gamma)
        end = time.time()
        if self.verbose:
            print(f'gram computation in {end - start} seconds')

        self.oneclass_signs = {}
        # this could run in parallel 
        for nm, cl in  self.base_models.items():
            # obtain base models predictions and scores 
            pred = np.sign(cl.predict(X))
            # score is +1 if base model is correct, -1 otherwise
            scores = 1 - np.abs(y.to_numpy()[:,0] - pred)
            scores[scores == 0] = -1
            if all([score == scores[0] for score in scores]):
                if self.verbose:
                    print(f'score for {nm} is always {scores[0]}')
                model = svm.OneClassSVM(kernel='precomputed')
                one_class = True
                self.oneclass_signs[nm] = scores[0]
            else:
                model =  svm.SVC(kernel="precomputed", C=self.C)
                one_class = False
                self.oneclass_signs[nm] = 1.0
            if self.param_grid is None or one_class:
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
            if one_class:
                self.smm_scores[nm] = 1.0
            else:
                self.smm_scores[nm] = model.score(gram, scores)
            self.base_scores[nm] = accuracy_score(y, pred) 
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

        # get all the base models predictions 
        self.base_predictions = {}
        for nm, cl in self.base_models.items():
            # obtain base model predictions 
            self.base_predictions[nm] = cl.predict(X)

        res = np.zeros(jX.shape[0])
        # compute weighted average, simple average, voting
        for nm, pr in self.base_predictions.items(): 
            model = self.base_models_classifiers[nm]
            # get classifier decision function
            df = model.decision_function(xnew) * self.oneclass_signs[nm] 
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
