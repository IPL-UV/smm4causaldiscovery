import cdt
from sklearn.metrics import accuracy_score
import time
import argparse
from smmw_ensemble import SMMwEnsemble
import numpy as np
from base_methods import fIGCI, fRECI


'''
function to run experiment over tuebingen data set
'''

def run(mech = 'nn', ntrain=100,
        size=100, noise_coeff=0.4,
        gamma = 100, rescale=True):


    gen = cdt.data.CausalPairGenerator(mech, noise_coeff=noise_coeff)
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
        gamma=gamma)
    
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
    Xt, yt = cdt.data.load_dataset("tuebingen")
  
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
