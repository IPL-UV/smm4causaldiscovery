import numpy as np
import pandas as pd

'''
Simple class for a naive ensamble 
of base bivariate causal discovery  methods
'''

class naive_ensamble():

    def __init__(self, base_models, strategy = 'voting'):
        self.strategy = strategy
        if type(base_models) is dict:
            self.base_models = base_models 
        else:
            raise Exception("""base_models argument should be 
                             a dictionary 
                             with base models""")
        

   
    def predict(self, X):
        # aggregate base models predictions 
        res = np.array([0 for i in range(X.shape[0])])
        for nm, cl in  self.base_models.items():
            pred = cl.predict(X)
            if self.strategy == 'voting':
                pred = np.sign(pred)
            res = res + pred 
        return(np.sign(res))

