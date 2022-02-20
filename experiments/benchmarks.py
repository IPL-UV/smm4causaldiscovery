import cdt
from sklearn.metrics import accuracy_score
import time
import argparse
from smmw_ensemble import SMMwEnsemble
import numpy as np
from base_methods import fIGCI, fRECI
from cdt.data import load_dataset
from .util import load_anlsmn, load_sim, noise_funcs

'''
function to run experiment over benchmarks data sets
'''

anlsmn=('AN', 'AN-s', 'LS', 'LS-s', 'MN-U')
sim = ('SIM', 'SIM-c', 'SIM-G', 'SIM-ln') 

benchmarks = { 'ANLSMN' : {'load' : load_anlsmn, 'names' : anlsmn},
        'SIM': {'load' : load_sim, 'names' : sim}}
        #'tuebingen': {'load': load_dataset , 'names': ('tuebingen',) }}

def run(mechs=('nn',), noises=('normal',),
        ncoeffs=(0.1,),
        ntrain=10, size=100):


    gen=cdt.data.CausalPairGenerator('linear', noise_coeff=0.4)
    X, y = gen.generate(1, npoints=size, rescale=True)
    for mech in mechs:
        for noise in noises:
            for ncoeff in ncoeffs:
                gen=cdt.data.CausalPairGenerator(mech, noise = noise_funcs[noise], 
                                                 noise_coeff=ncoeff)
                X1, y1 = gen.generate(ntrain, npoints=size, rescale=True)
                X=X.append(X1)
                y=y.append(y1)
    
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
        parallel=False,
        verbose=True,
        gamma='median')
    
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
            Xt, yt = load(name)
  
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
