from experiments import generated_data 
from experiments import util 
from itertools import product
import os


gamma = 100
nrep = 10
mechs = ('nn', 'gp_add', 'gp_mix', 'polynomial', 'sigmoid_add', 'sigmoid_mix')
ntrains = (100,)
ntests = (100,)
sizes = (50, 100, 250, 500, 750, 1000)
ncoeffs = (0.4,)

exp_set = product(mechs, sizes, ntrains, ntests, ncoeffs)
for (mech, size, ntrain, ntest, ncoeff) in exp_set:
    path = os.path.join('results', 'generated_data', 
            f'{mech}{ncoeff}_s{size}_ntrain{ntrain}_ntest{ntest}')
    os.makedirs(path, exist_ok=True)
    for i in range(nrep):
        res = generated_data.run(mech, ntrain, ntest, size, ncoeff, gamma, True) 
        util.save_csv(res, os.path.join(path, f'rep{i}.csv'))
