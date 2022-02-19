import csv
import pandas as pd
import cdt 
import numpy as np
from os import path

rng = np.random.default_rng(2022)

def normal(points):
    """Init a noise variable."""
    return rng.standard_normal((points,1))

def uniform(points):
    return rng.uniform(0,1,(points,1)) 


noise_funcs = {'normal': normal, 'uniform': uniform }

def save_csv2(data, path):
    pd.DataFrame.from_dict(data).to_csv(path)



def save_csv(data, path):
    csv_file = path
    csv_columns = list(data[0].keys()) 
    
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for d in data:
                writer.writerow(d)

    except IOError:
        print("I/O error")

def load_data(pairs, target):
    X = cdt.utils.io.read_causal_pairs(args.pairs, scale=args.rescale)
    y = pd.read_csv(args.target).set_index('SampleID')
    return X, y
    
def data_train_test(X, y, ntrain, ntest):
    Xtrain = X.iloc[0:ntrain, :]
    Xtest = X.iloc[ntrain:(ntrain+ntest), :]
    
    ytrain = y.iloc[0:ntrain, :]
    ytest = y.iloc[ntrain:(ntrain+ntest), :]
    return Xtrain, ytrain, Xtest, ytest


def load_anlsmn(name = 'AN', rescale=True): 
    bp = path.join('data/ANLSMN_pairs/', name)
    data = []
    for i in range(100):
        pair = pd.read_csv(path.join(bp, f'pair_{i+1}.txt'))
        pair.columns = ['xx', 'A', 'B']
        A = pair.A.to_numpy()
        B = pair.B.to_numpy()
        if rescale:
            A = (A - np.mean(A)) / np.std(A)
            B = (B - np.mean(B)) / np.std(B)
        data.append((i+1, A, B))
        
    X = pd.DataFrame(data, columns=['SampleID', 'A', 'B']) 
    X = X.set_index('SampleID')

    y = pd.read_csv(path.join(bp, 'pairs_gt.txt'), header=None)
    y[y==0] = -1

    return X, y

def load_sim(name = 'SIM', rescale=True): 
    bp = path.join('data/SIM_pairs/', name)
    data = []
    for i in range(100):
        pair = pd.read_table(path.join(bp, f'pair{i+1:04}.txt'),
                header=None, sep='\s+', engine='python')
        pair.columns = ['A', 'B']
        A = pair.A.to_numpy()
        B = pair.B.to_numpy()
        if rescale:
            A = (A - np.mean(A)) / np.std(A)
            B = (B - np.mean(B)) / np.std(B)
        data.append((i+1, A, B))
        
    X = pd.DataFrame(data, columns=['SampleID', 'A', 'B']) 
    X = X.set_index('SampleID')

    tmp = pd.read_table(path.join(bp, 'pairmeta.txt'),
            header=None, sep='\s', engine='python')
    tmp.columns = ['id', 'c1', 'c2', 'e1', 'e2', 'w']
    
    y = tmp.c1 
    y[y==2] = -1

    return X, y
