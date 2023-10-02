import cdt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import argparse
from smmw_ensemble import SMMwEnsemble
import numpy as np
from base_methods import fIGCI, fRECI, fastANM, fastBV
from cdt.data.loader import load_tuebingen
from .util import load_anlsmn, load_sim, noise_funcs

'''
function to run experiment over benchmarks data sets
'''


def run():

     # load tuebingen shuffled
    Xall, yall = load_tuebingen(shuffle=True)
#    for i in range(Xall.shape[0]):
#        a = Xall.iloc[i,0]
#        a = (a - a.mean()) / a.std()
#        Xall.iloc[i,0] = a
#        b = Xall.iloc[i,1]
#        b = (b - b.mean()) / b.std()
#        Xall.iloc[i,1] = b

    X, Xt, y, yt = train_test_split(Xall, yall, test_size = 0.5) 


    train_time = {} 
    print('start meta causal')
    start = time.time()
    model = SMMwEnsemble({
        "CDS" : cdt.causality.pairwise.CDS(),
        "fastANM": fastANM(),
        'fastBV': fastBV(),
        #"ANM" : cdt.causality.pairwise.ANM(), 
        #"BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
        "IGCI" : fIGCI(), 
        "RECI": fRECI()},
        include_constant=False,
        exp_weights=False,
        param_grid = {"C": np.logspace(-4, 8, 50)},
        #C = 0.01,
        size = 500,
        parallel=True,
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
    rcc = cdt.causality.pairwise.RCC(rand_coeff=1000)
    rcc.fit(X,y)
    end = time.time()
    train_time['rcc'] = end - start
    print(f'rcc fitted in {end-start} seconds') 


    print('test')
    allscores = {}
    test_time = {}

    # smm
    start = time.time()
    smm_score = model.score(Xt,yt) 
    model.parallel = True
    end = time.time() 
    test_time['smm_ensemble'] = end - start

    # jarfo 
    start = time.time()
    jarfo_score = accuracy_score(yt, np.sign(jarfo.predict(Xt))) 
    end = time.time() 
    test_time['jarfo'] = end - start

    #rcc
    start = time.time()
    pred = np.sign(rcc.predict(Xt))
    rcc_score = accuracy_score(yt, pred)
    print(f'rcc score {rcc_score}')
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
    allscores.update({'tuebingen' : scores})

    return allscores
