import cdt
from sklearn.metrics import accuracy_score
import time
import argparse
from .util import save_csv, save_csv2
from smmw_ensemble import SMMwEnsemble
import numpy as np
from base_methods import fIGCI, fRECI
from .util import noise_funcs


'''
function to run experiment over generated data
'''



def run(rep, mechs = ('nn',), noises = ('normal', 'uniform'),
        ncoeffs = (0.1,),   
        ntrain=10, ntest=1000, size=100, rescale=True):

    gen=cdt.data.CausalPairGenerator('linear')
    X, y = gen.generate(1, npoints=size, rescale=True)
    for mech in mechs:
        for noise in noises:
            for ncoeff in ncoeffs:
                gen=cdt.data.CausalPairGenerator(mech, noise=noise_funcs[noise],
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
        parallel=True,
        njobs=5,
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
    rcc = cdt.causality.pairwise.RCC()
    rcc.fit(X,y)
    end = time.time()
    train_time['rcc'] = end - start
    print(f'rcc fitted in {end-start} seconds') 


    # testing
    for mech in mechs:
        for noise in noises:
            for ncoeff in ncoeffs:
                path = os.path.join('results', 'generated_data_mix', 
                       f'{mech}_{noise}{ncoeff}_s{size}_ntrain{ntrain}_ntest{ntest}_gamma{gamma}')
                os.makedirs(path, exist_ok=True)

                gen=cdt.data.CausalPairGenerator(mech, noise=noise_funcs[noise],
                                                 noise_coeff=ncoeff)
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
                util.save_csv((scores, train_time, test_time), os.path.join(path, f'rep{rep}.csv'))
                util.save_csv2(model.smms_df, os.path.join(path, f'df_rep{rep}.csv'))
