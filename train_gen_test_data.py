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
parser.add_argument('-s', '--size', dest='size', action='store',
                    type = int,
                    default = 10, 
                    help='sample size, number of points')

args = parser.parse_args()
print(args)

gen = cdt.data.CausalPairGenerator(args.mech)
X, y = gen.generate(args.ntrain, npoints = args.size, rescale = args.rescale)


print('start meta causal')
start = time.process_time()
model = meta_causal_smm({
                        "CDS" : cdt.causality.pairwise.CDS(),
                         #"ANM" : cdt.causality.pairwise.ANM(), 
                         "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
                         "IGCI" : cdt.causality.pairwise.IGCI(), 
                        "RECI": cdt.causality.pairwise.RECI()},
                        kernel = lambda a,b: rbf_kernel(a, b, 1), C = 10)

model.fit(X,y) 
end = time.process_time() 
print(f'meta smm model fitted in {end-start} seconds')


rcc = cdt.causality.pairwise.RCC(njobs = 1)
rcc.fit(X, y) 


voting = naive_ensamble({
                        "CDS" : cdt.causality.pairwise.CDS(),
                         #"ANM" : cdt.causality.pairwise.ANM(), 
                         "IGCI" : cdt.causality.pairwise.IGCI(), 
                         "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
                        "RECI": cdt.causality.pairwise.RECI()})

averaging = naive_ensamble({
                        "CDS" : cdt.causality.pairwise.CDS(),
                         #"ANM" : cdt.causality.pairwise.ANM(), 
                         "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
                         "IGCI" : cdt.causality.pairwise.IGCI(), 
                        "RECI": cdt.causality.pairwise.RECI()}, strategy = '')

### testing
Xt, yt = cdt.data.load_dataset("tuebingen") 


methods = { 
        "CDS" : cdt.causality.pairwise.CDS(), 
        #"ANM" :cdt.causality.pairwise.ANM(), 
        "BivariateFit" : cdt.causality.pairwise.BivariateFit(), 
        "IGCI" : cdt.causality.pairwise.IGCI(),
        "RECI": cdt.causality.pairwise.RECI(),
        'meta': model,
        #'voting': voting,
        'rcc': rcc,
        'averaging': averaging}


yt = yt.to_numpy()[:,0]
acc = {}
for nm,mth in methods.items():
    pr = mth.predict(Xt) 
    acc[nm] = accuracy_score(yt, np.sign(pr))
    print(f"{nm}: {acc[nm]}") 
