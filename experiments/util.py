import csv
import pandas as pd
import cdt 
import numpy as np


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
    return X,y
    
def data_train_test(X, y, ntrain, ntest):
    Xtrain = X.iloc[0:ntrain, :]
    Xtest = X.iloc[ntrain:(ntrain+ntest), :]
    
    ytrain = y.iloc[0:ntrain, :]
    ytest = y.iloc[ntrain:(ntrain+ntest), :]
    return Xtrain, ytrain, Xtest, ytest
