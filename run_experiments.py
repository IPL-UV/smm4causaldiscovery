from experiments import generated_data 
from experiments import util 
from itertools import product
import os
import argparse

parser = argparse.ArgumentParser( )
parser.add_argument('--generated1', action='store_true',
                    help='run generated data experiment 1')

parser.add_argument('--generated2', action='store_true',
                    help='run generated data experiment 2')

args = parser.parse_args()
print(args)



if args.generated1:
    gamma = 1
    nrep = 10
    mechs = ('nn', 'polynomail', 'sigmoid_add', 'sigmoid_mix',)
    ntrains = (100,)
    ntests = (100,)
    sizes = (50, 100, 250, 500, 750, 1000)
    ncoeffs = (0.4, 0.6)
    
    exp_set = product(mechs, ncoeffs, sizes, ntrains, ntests)
    for (mech, ncoeff, size, ntrain, ntest) in exp_set:
        path = os.path.join('results', 'generated_data', 
                f'{mech}{ncoeff}_s{size}_ntrain{ntrain}_ntest{ntest}_gamma{gamma}')
        os.makedirs(path, exist_ok=True)
        for i in range(nrep):
            res = generated_data.run(mech, ntrain, ntest, size, gamma, size, True) 
            util.save_csv(res, os.path.join(path, f'rep{i}.csv'))


if args.generated2:
    gamma = 1
    nrep = 10
    mechs = ('nn', 'polynomail', 'sigmoid_add', 'sigmoid_mix',)
    ntrains = (100, 250, 500, 750, 1000)
    ntests = (1000,)
    sizes = (250,)
    ncoeffs = (0.2, 0.4, 0.6)
    
    exp_set = product(mechs, ncoeffs, sizes, ntrains, ntests)
    for (mech, ncoeff, size, ntrain, ntest) in exp_set:
        path = os.path.join('results', 'generated_data', 
                f'{mech}{ncoeff}_s{size}_ntrain{ntrain}_ntest{ntest}_gamma{gamma}')
        os.makedirs(path, exist_ok=True)
        for i in range(nrep):
            res = generated_data.run(mech, ntrain, ntest, size, gamma, size, True) 
            util.save_csv(res, os.path.join(path, f'rep{i}.csv'))
