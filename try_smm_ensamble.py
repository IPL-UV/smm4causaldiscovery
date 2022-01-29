import cdt
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import * 
import numpy as np
import pandas as pd
import time
import argparse
from smm_ensamble import SMMEnsamble 
import base_methods

parser = argparse.ArgumentParser( )
parser.add_argument('--rescale', dest='rescale', action='store_true',
                    help='rescale oprion')
parser.add_argument('--mech', dest='mech', action='store',
                    default = 'linear',  
                    help='sampling causal mechanism')
parser.add_argument('--ntrain', dest='ntrain', action='store',
                    type = int, 
                    default = 100,
                    help='number of pairs in the training set')
parser.add_argument('--ntest', dest='ntest', action='store',
                    type = int,
                    default = 10,
                    help='number of pairs in the test set')
parser.add_argument('-s', '--size', dest='size', action='store',
                    type = int,
                    default = 50, 
                    help='sample size, number of points')
parser.add_argument('-g', '--gamma', dest='gamma', action='store',
                    type = float,
                    default = 1, 
                    help='gamma Gaussain RBF')

args = parser.parse_args()
print(args)

gen = cdt.data.CausalPairGenerator(args.mech)
X, y = gen.generate(args.ntrain, npoints = args.size, rescale = args.rescale)


print('start meta causal')
start = time.process_time()
model = SMMEnsamble({
                        "CDS" : cdt.causality.pairwise.CDS(),
                         "ANM" : cdt.causality.pairwise.ANM(), 
                         "IGCI" : base_methods.fIGCI(), 
                        "RECI": cdt.causality.pairwise.RECI()},
                        param_grid = {"C": np.logspace(-2, 3, 20)},
                        gamma = args.gamma, 
                        verbose = True)

model.fit(X,y) 
end = time.process_time() 
print(f'meta smm model fitted in {end-start} seconds')


### testing
Xt, yt = gen.generate(args.ntest, npoints = args.size, rescale = args.rescale)


score = model.score(Xt, yt.to_numpy()[:,0])
scores_alternatives  = model.score_alternatives(yt.to_numpy()[:,0]) 
scores_base = model.score_base(yt.to_numpy()[:,0])


print(f'score smm-weighted ensamble: {score}')

print(scores_alternatives)

print(scores_base)