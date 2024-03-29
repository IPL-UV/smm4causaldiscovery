from experiments import tuebingen2, tuebingen, mixgenerated, generated_data, benchmarks 
from experiments import util 
from itertools import product
import os
import argparse

parser = argparse.ArgumentParser( )
parser.add_argument('--generated1', action='store_true',
                    help='run generated data experiment 1')

parser.add_argument('--generated2', action='store_true',
                    help='run generated data experiment 2')

parser.add_argument('--benchmarks', action='store_true',
                    help='run experiment on benchmarks')

parser.add_argument('--mixgenerated', action='store_true',
                    help='run generated data experiment 2')

parser.add_argument('--tuebingen', action='store_true',
                    help='run generated tuebingen exp')

parser.add_argument('--tuebingen2', action='store_true',
                    help='run generated tuebingen exp')



args = parser.parse_args()
print(args)


if args.generated1:
    nrep = 10
    mechs = ('nn', 'polynomial', 'sigmoid_add', 'sigmoid_mix', 'gp_add', 'gp_mix')
    ntrains = (100,)
    ntests = (100,)
    sizes = (50, 100, 250, 500, 750, 1000)
    noises = ('normal2', 'uniform2')
    ncoeffs = (0.5,)
    
    exp_set = product(mechs, noises, ncoeffs, sizes, ntrains, ntests)
    for (mech, noise, ncoeff, size, ntrain, ntest) in exp_set:
        path = os.path.join('results', 'generated_data', 
                f'{mech}{noise}{ncoeff}_s{size}_ntrain{ntrain}_ntest{ntest}')
        os.makedirs(path, exist_ok=True)
        for i in range(nrep):
            res = generated_data.run(mech=mech,
                                     noise=noise,
                                     ncoeff=ncoeff,
                                     ntrain=ntrain,
                                     ntest=ntest,
                                     size=size,
                                     rescale=True) 
            util.save_csv(res[0:-1], os.path.join(path, f'rep{i}.csv'))
            util.save_csv2(res[-1], os.path.join(path, f'df_rep{i}.csv'))


if args.mixgenerated:
    nrep = 5
    ntrains = (750, 1000)
    ntest = 1000
    size = 250
    mechs = ('gp_add', 'gp_mix')
    noises = ('normal2',)
    ncoeffs = (0.5,)
    
    for ntrain in ntrains:
        for i in range(nrep):
            mixgenerated.run(i, mechs=mechs,
                             noises=noises,
                             ncoeffs=ncoeffs,
                             ntrain=ntrain,
                             ntest=ntest,
                             size=size,
                             rescale=True) 

if args.benchmarks: 
    nrep = 5
    path = os.path.join('results', 'benchmarks')
    os.makedirs(path, exist_ok=True)
    for i in range(nrep):
        res, dfs = benchmarks.run() 
        util.save_csv2(res, os.path.join(path, f'rep{i}.csv'))
        for key, df in dfs.items():
            util.save_csv2(df, os.path.join(path, f'df_rep{i}_{key}.csv'))


if args.tuebingen: 
    nrep = 10
    ## effective training size is ntrain * |mechs|*|noises|*|ncoeffs| 
    ntrains = (500,) 
    sizes = (250,)
    mechs = ('nn', )
    noises = ('normal2', 'uniform2')
    ncoeffs = (0.5,)
    
    exp_set = product(sizes, ntrains)
    for (size, ntrain) in exp_set:
        path = os.path.join('results', 'tuebingen', 
                f'mix_s{size}_ntrain{ntrain}')
        os.makedirs(path, exist_ok=True)
        for i in range(nrep):
            res = tuebingen.run(mechs=mechs,
                                 noises=noises,
                                 ncoeffs=ncoeffs,
                                 ntrain=ntrain,
                                 size=size) 
            util.save_csv2(res, os.path.join(path, f'rep{i}.csv'))


if args.tuebingen2: 
    nrep = 20
    path = os.path.join('results', 'tuebingen')
    os.makedirs(path, exist_ok=True)
    for i in range(nrep):
        res = tuebingen2.run() 
        util.save_csv2(res, os.path.join(path, f'rep{i}.csv'))
