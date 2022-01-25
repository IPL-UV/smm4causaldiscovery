import cdt
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import * 
import numpy as np
import pandas as pd
import time
from naive_ensamble import naive_ensamble
import argparse
import csv
from jax_smm_ensamble import SMMEnsamble   
from utilities import cdt_data_to_numpy

'''
script to run experiment over generated data
'''

parser = argparse.ArgumentParser( )
parser.add_argument('--rescale', dest='rescale', action='store_true',
                    help='rescale option')
parser.add_argument('--mech', dest='mech', action='store',
                    default = 'linear',  
                    help='sampling causal mechanism')
parser.add_argument('--ntrain', dest='ntrain', action='store',
                    type = int, 
                    default = 10,
                    help='number of pairs in the training set')
parser.add_argument('--ntest', dest='ntest', action='store',
                    type = int,
                    default = 10,
                    help='number of pairs in the test set')
parser.add_argument('-s', '--size', dest='size', action='store',
                    type = int,
                    default = 10, 
                    help='sample size, number of points')
parser.add_argument('--noise-coeff', dest='noise_coeff', action='store',
                    type = float,
                    default = 0.4, 
                    help='noise coefficient')
parser.add_argument('-o', '--output', dest='output', action='store',
                    type = str,
                    default = 'out.csv', 
                    help='name of output csv file')


args = parser.parse_args()
print(args)

gen = cdt.data.CausalPairGenerator(args.mech, noise_coeff = args.noise_coeff)
X, y = gen.generate(args.ntrain, npoints = args.size, rescale = args.rescale)


train_time = {} 
print('start meta causal')
start = time.time()
model = SMMEnsamble({
    "CDS" : cdt.causality.pairwise.CDS(),
    "ANM" : cdt.causality.pairwise.ANM(), 
    "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
    "IGCI" : cdt.causality.pairwise.IGCI(), 
    "RECI": cdt.causality.pairwise.RECI()},
    param_grid = {"C": np.logspace(-2, 3, 20)},
    verbose = True,
    gamma = 1)

model.fit(X, y) 
end = time.time() 
train_time['meta'] = end - start
print(f'meta smm model fitted in {end-start} seconds')

averaging = naive_ensamble({
    "CDS" : cdt.causality.pairwise.CDS(),
    "ANM" : cdt.causality.pairwise.ANM(), 
    "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
    "IGCI" : cdt.causality.pairwise.IGCI(), 
    "RECI": cdt.causality.pairwise.RECI()}, strategy = '')

start = time.time()
jarfo = cdt.causality.pairwise.Jarfo()
jarfo.fit(X, y) 
end = time.time()
train_time['jarfo'] = end - start
print('jarfo fitted') 

start = time.time()
rcc = cdt.causality.pairwise.RCC(njobs = 1)
rcc.fit(X, y) 
end = time.time()
train_time['rcc'] = end - start
print('rcc fitted') 


### testing
Xt, yt = gen.generate(args.ntest, npoints = args.size, rescale = args.rescale)


methods = { 
        "CDS" : cdt.causality.pairwise.CDS(), 
        "ANM" : cdt.causality.pairwise.ANM(), 
        "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
        "IGCI" : cdt.causality.pairwise.IGCI(),
        "RECI": cdt.causality.pairwise.RECI(),
        'meta': model,
        #'voting': voting,
        'averaging': averaging,
        'jarfo': jarfo,
        'rcc': rcc}


test_time = {} 
yt = yt.to_numpy()[:,0]
acc = {}
for nm,mth in methods.items():
    start = time.time()
    pr = mth.predict(Xt) 
    end = time.time()
    test_time[nm] = end - start
    acc[nm] = accuracy_score(yt, np.sign(pr))
    print(f"{nm}: {acc[nm]}") 

exit()
print("save results to csv")
csv_file = args.output
csv_columns = list(methods.keys()) 

try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(acc)
        writer.writerow(train_time)
        writer.writerow(test_time)

except IOError:
    print("I/O error")
