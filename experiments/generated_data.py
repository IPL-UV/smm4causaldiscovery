import cdt
from sklearn.metrics import accuracy_score
import time
import argparse
from .util import save_csv
from smm_ensamble import SMMEnsamble
import numpy as np


'''
script to run experiment over generated data
'''

def run(mech='nn', ntrain=100, ntest=100, size=100, noise_coeff=0.4, rescale=True):

    gen = cdt.data.CausalPairGenerator(mech, noise_coeff=noise_coeff)
    X, y = gen.generate(ntrain, npoints=size, rescale=rescale)
    
    train_time = {} 
    print('start meta causal')
    start = time.time()
    model = SMMEnsamble({
        "CDS" : cdt.causality.pairwise.CDS(),
        "ANM" : cdt.causality.pairwise.ANM(), 
        "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
        "IGCI" : cdt.causality.pairwise.IGCI(), 
        "RECI": cdt.causality.pairwise.RECI()},
        param_grid = {"C": np.linspace(1e-1, 1e3, 20)},
        verbose = True,
        gamma = 100)
    
    model.fit(X, y) 
    end = time.time() 
    train_time['meta'] = end - start
    print(f'meta smm model fitted in {end-start} seconds')

    ## testing
    Xt, yt = gen.generate(ntest, npoints=size, rescale=rescale)

    scores = {'smm_ensamble': model.score(Xt, yt.to_numpy()[:,0])}
    scores.update(model.score_alternatives(yt.to_numpy()[:,0]))
    scores.update(model.score_base(yt.to_numpy()[:,0]))

    return scores
