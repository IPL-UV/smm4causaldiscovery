from experiments import generated_data, benchmarks 
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



args = parser.parse_args()
print(args)


if args.generated1:
    gamma = 1
    nrep = 10
    mechs = ('nn', 'polynomial', 'sigmoid_add', 'sigmoid_mix', 'gp_add', 'gp_mix')
    ntrains = (100,)
    ntests = (100,)
    sizes = (50, 100, 250, 500, 750, 1000)
    ncoeffs = (0.2, 0.4, 0.6, 1.0)
    
    exp_set = product(mechs, ncoeffs, sizes, ntrains, ntests)
    for (mech, ncoeff, size, ntrain, ntest) in exp_set:
        path = os.path.join('results', 'generated_data', 
                f'{mech}{ncoeff}_s{size}_ntrain{ntrain}_ntest{ntest}_gamma{gamma}')
        os.makedirs(path, exist_ok=True)
        for i in range(nrep):
            res = generated_data.run(mech, ntrain, ntest, size, ncoeff, gamma, True) 
            util.save_csv(res[0:-1], os.path.join(path, f'rep{i}.csv'))
            util.save_csv2(res[-1], os.path.join(path, f'df_rep{i}.csv'))


if args.generated2:
    gamma = 1
    nrep = 10
    mechs = ('nn', 'polynomial', 'sigmoid_add', 'sigmoid_mix', 'gp_add', 'gp_mix')
    ntrains = (100, 250, 500, 750, 1000)
    ntests = (1000,)
    sizes = (250,)
    ncoeffs = (0.4,)
    
    exp_set = product(mechs, ncoeffs, sizes, ntrains, ntests)
    for (mech, ncoeff, size, ntrain, ntest) in exp_set:
        path = os.path.join('results', 'generated_data', 
                f'{mech}{ncoeff}_s{size}_ntrain{ntrain}_ntest{ntest}_gamma{gamma}')
        os.makedirs(path, exist_ok=True)
        for i in range(nrep):
            res = generated_data.run(mech, ntrain, ntest, size, ncoeff, gamma, True) 
            util.save_csv(res[0:-1], os.path.join(path, f'rep{i}.csv'))
            util.save_csv2(res[-1], os.path.join(path, f'df_rep{i}.csv'))


if args.benchmarks: 
    gamma = 1
    nrep = 10
    ntrains = (20,)
    sizes = (250,)
    
    exp_set = product(sizes, ntrains)
    for (size, ntrain) in exp_set:
        path = os.path.join('results', 'benchmarks', 
                f'mix_s{size}_ntrain{ntrain}_gamma{gamma}')
        os.makedirs(path, exist_ok=True)
        for i in range(nrep):
            res = benchmarks.run(ntrain, size, gamma) 
            util.save_csv2(res, os.path.join(path, f'rep{i}.csv'))
