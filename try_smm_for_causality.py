import cdt
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import * 
from sklearn import svm
import numpy as np
import pandas as pd
import time
from smm import smm
import argparse

parser = argparse.ArgumentParser( )
parser.add_argument('--rescale', dest='rescale', action='store_true',
                    help='rescale oprion')
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
parser.add_argument('-C' , dest='C', action='store',
                    type = float,
                    default = 10, 
                    help='cost SVC')
parser.add_argument('-g', '--gamma', dest='gamma', action='store',
                    type = float,
                    default = 1, 
                    help='gamma Gaussain RBF')

args = parser.parse_args()
print(args)

gen = cdt.data.CausalPairGenerator(args.mech)
X, y = gen.generate(args.ntrain, npoints = args.size, rescale = args.rescale)


model = smm(kernel = lambda a,b: rbf_kernel(a, b, args.gamma), 
        base_svm = svm.SVC,
        normalize = True,
        C = args.C)
model.fit(x = X.to_numpy(), y = y.to_numpy()[:,0])


### testing
Xt, yt = gen.generate(args.ntest, npoints = args.size, rescale = args.rescale)


methods = { 
        "CDS" : cdt.causality.pairwise.CDS(), 
        "ANM" :cdt.causality.pairwise.ANM(), 
        "IGCI" : cdt.causality.pairwise.IGCI(),
        "RECI": cdt.causality.pairwise.RECI()
        }


acc = {}
for nm,mth in methods.items():
    pr = mth.predict(Xt) 
    acc[nm] = accuracy_score(yt.to_numpy()[:,0], np.sign(pr))
    print(f"{nm}: {acc[nm]}") 



print(model.score(Xt.to_numpy(), yt.to_numpy()[:,0]))

