import cdt
from sklearn.metrics import accuracy_score
import time
import argparse
from .util import save_csv
from smmw_ensemble import SMMwEnsemble
import numpy as np
from base_methods import fIGCI, fRECI
import os
from .util import noise_funcs


'''
function to run experiment over generated data
'''

def run(mech='nn', noise='normal2', ncoeff=0.5, ntrain=100, ntest=100, size=100, rescale=True):

    gen = cdt.data.CausalPairGenerator(mech, noise=noise_funcs[noise], noise_coeff=ncoeff)
    X, y = gen.generate(ntrain, npoints=size, rescale=rescale)
    
    train_time = {} 
    print('start meta causal')
    start = time.time()
    model = SMMwEnsemble({
        "CDS" : cdt.causality.pairwise.CDS(),
        "ANM" : cdt.causality.pairwise.ANM(), 
        "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
        "IGCI" : fIGCI(), 
        "RECI": fRECI()},
        include_constant=False,
        exp_weights=False,
        param_grid = {"C": np.logspace(-3, 5, 20)},
        parallel=True,
        njobs=5,
        verbose=True,
        gamma=0.5)
    
    model.fit(X, y) 
    end = time.time() 
    train_time['smm_ensemble'] = end - start
    print(f'smm-w ensemble fitted in {end-start} seconds')

    # fit jarfo 
    start = time.time()
    jarfo = cdt.causality.pairwise.Jarfo()
    jarfo.fit(X,y) 
    end = time.time()
    train_time['jarfo'] = end - start
    print(f'jarfo fitted in {end-start} seconds') 

    #fit rcc 
    start = time.time()
    rcc = cdt.causality.pairwise.RCC()
    rcc.fit(X,y)
    end = time.time()
    train_time['rcc'] = end - start
    print(f'rcc fitted in {end-start} seconds') 

    # testing
    Xt, yt = gen.generate(ntest, npoints=size, rescale=rescale)
  
    test_time = {}

    # smm
    start = time.time()
    smm_score = model.score(Xt,yt) 
    end = time.time() 
    test_time['smm_ensemble'] = end - start

    # jarfo 
    start = time.time()
    jarfo_score = accuracy_score(yt, np.sign(jarfo.predict(Xt))) 
    end = time.time() 
    test_time['jarfo'] = end - start

    #rcc
    start = time.time()
    rcc_score = accuracy_score(yt, np.sign(rcc.predict(Xt))) 
    end = time.time() 
    test_time['rcc'] = end - start

    #get all scores 
    scores = {'smm_ensemble': smm_score, 
              "jarfo": jarfo_score,
              "rcc": rcc_score
              }

    # add scores of alternative ensembles
    scores.update(model.score_alternatives(yt))
    # add scores of base methods
    scores.update(model.score_base(yt))

    return (scores, train_time, test_time, model.smms_df)
