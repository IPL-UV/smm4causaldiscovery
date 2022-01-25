import cdt
from meta_causal_smm import meta_causal_smm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import * 
import numpy as np
import pandas as pd
import time
from naive_ensamble import naive_ensamble
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


print('start meta causal')
start = time.process_time()
model = meta_causal_smm({
                        "CDS" : cdt.causality.pairwise.CDS(),
                         "ANM" : cdt.causality.pairwise.ANM(), 
                         "IGCI" : cdt.causality.pairwise.IGCI(), 
                        "RECI": cdt.causality.pairwise.RECI()},
                        kernel = lambda a,b: rbf_kernel(a, b, gamma = 1), 
                        #kernel = lambda a,b: rbf_kernel(a, b, args.gamma), 
                        verbose = True,  C = args.C)

model.fit(X,y) 
end = time.process_time() 
print(f'meta smm model fitted in {end-start} seconds')



voting = naive_ensamble({
                        "CDS" : cdt.causality.pairwise.CDS(),
                         "ANM" : cdt.causality.pairwise.ANM(), 
                         "IGCI" : cdt.causality.pairwise.IGCI(), 
                        "RECI": cdt.causality.pairwise.RECI()})

averaging = naive_ensamble({
                        "CDS" : cdt.causality.pairwise.CDS(),
                         "ANM" : cdt.causality.pairwise.ANM(), 
                         "IGCI" : cdt.causality.pairwise.IGCI(), 
                        "RECI": cdt.causality.pairwise.RECI()}, strategy = '')

### testing
Xt, yt = gen.generate(args.ntest, npoints = args.size, rescale = args.rescale)


methods = { 
        "CDS" : cdt.causality.pairwise.CDS(), 
        "ANM" :cdt.causality.pairwise.ANM(), 
        "IGCI" : cdt.causality.pairwise.IGCI(),
        "RECI": cdt.causality.pairwise.RECI(),
        'meta': model,
        'voting': voting,
        'averaging': averaging}


acc = {}
for nm,mth in methods.items():
    pr = mth.predict(Xt) 
    acc[nm] = accuracy_score(yt.to_numpy()[:,0], np.sign(pr))
    print(f"{nm}: {acc[nm]}") 



model.score(Xt, yt)

