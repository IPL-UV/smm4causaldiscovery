import cdt
from sklearn.metrics import accuracy_score
import time
import argparse
from .util import save_csv
from smm_ensamble import SMMEnsamble
import numpy as np
from base_methods import fIGCI


'''
function to run experiment over generated data
'''

def run(mech='nn', ntrain=100, ntest=100, size=100, noise_coeff=0.4, gamma = 100, rescale=True):

    gen = cdt.data.CausalPairGenerator(mech, noise_coeff=noise_coeff)
    X, y = gen.generate(ntrain, npoints=size, rescale=rescale)
    
    train_time = {} 
    print('start meta causal')
    start = time.time()
    model = SMMEnsamble({
        "CDS" : cdt.causality.pairwise.CDS(),
        "ANM" : cdt.causality.pairwise.ANM(), 
        "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
        "IGCI" : fIGCI(), 
        "RECI": cdt.causality.pairwise.RECI()},
        param_grid = {"C": np.linspace(1e-1, 1e3, 20)},
        verbose = True,
        gamma = gamma)
    
    model.fit(X, y) 
    end = time.time() 
    train_time['smm_ensamble'] = end - start
    print(f'smm enamblel fitted in {end-start} seconds')

    # fit jarfo 
    start = time.time()
    jarfo = cdt.causality.pairwise.Jarfo()
    jarfo.fit(X,y) 
    end = time.time()
    train_time['jarfo'] = end - start
    print(f'jarfo fitted in {end-start} second') 

    #fit rcc 
    start = time.time()
    rcc = cdt.causality.pairwise.RCC()
    rcc.fit(X,y)
    end = time.time()
    train_time['rcc'] = end - start
    print(f'rcc fitted in {end-start} second') 

    # testing
    Xt, yt = gen.generate(ntest, npoints=size, rescale=rescale)
  
    test_time = {}

    # smm
    start = time.time()
    smm_score = model.score(Xt,yt) 
    end = time.time() 
    test_time['smm_ensamble'] = end - start

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
    scores = {'smm_ensamble': smm_score, 
              "jarfo": jarfo_score,
              "rcc": rcc_score
              }

    # add scores of alternative ensambles
    scores.update(model.score_alternatives(yt))
    # add scores of base methods
    scores.update(model.score_base(yt))

    return (scores, train_time, test_time)
