import cdt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import argparse
from smmw_ensemble import SMMwEnsemble
import numpy as np
from base_methods import fIGCI, fRECI
from cdt.data import load_dataset
from .util import load_anlsmn, load_sim, noise_funcs
import pandas as pd

'''
function to run experiment over benchmarks data sets
'''

anlsmn=('AN', 'AN-s', 'LS', 'LS-s', 'MN-U')
sim = ('SIM', 'SIM-c', 'SIM-G', 'SIM-ln') 

benchmarks = { 'ANLSMN' : {'load' : load_anlsmn, 'names' : anlsmn},
        'SIM': {'load' : load_sim, 'names' : sim}}
        #'tuebingen': {'load': load_dataset , 'names': ('tuebingen',) }}

def run():

    Xtrain_all = []
    ytrain_all = []
    test = {}
    for key, bench in benchmarks.items():
        load = bench['load']
        names = bench['names']
        for name in names:
            X, y = load(name)
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.5)
            Xtrain_all += (Xtrain,)
            ytrain_all += (ytrain,)
            test[name] = { 'X':Xtest, 'y': ytest}

    print(len(Xtrain_all))
    X = pd.concat(Xtrain_all)
    print(X.shape)
    y = pd.concat(ytrain_all)
    print(y.shape)
    
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
        verbose=True,
        gamma=100)
    
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

    allscores = {}
    # testing
    for key, bench in benchmarks.items():
        load = bench['load']
        names = bench['names']
        for name in names:
            Xt = test[name]['X']
            yt = test[name]['y']
  
            test_time = {}

            # smm
            start = time.time()
            if name == 'tuebingen':
                model.parallel=False
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
            allscores.update({name : scores})

    return allscores
