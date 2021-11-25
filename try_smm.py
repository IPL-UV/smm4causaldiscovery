import pandas as pd
from cdt import data
from sklearn.metrics.pairwise import * 
from sklearn import svm
from smm import *
import random

'''
univariate case
For label = 1, X ~ N(5,1)
For label = 2, X ~ N(1,1)
'''
print('Univariate example with X~N(m,1), m = 5 for label = 1 and m = 1 for label = 2') 
Y = np.array([1,1,2,2,2], dtype=np.int32)
X = np.array([
       np.random.normal(5,1, 100),         
       np.random.normal(5,1, 100),         
       np.random.normal(1,1, 100),         
       np.random.normal(1,1,100),         
       np.random.normal(1,1, 100),         
       ])


print('smm with rbf_kernel(1)')
model = smm(kernel = lambda a,b: rbf_kernel(a, b, 1), normalize = True)
model.fit(x = X, y = Y)

print("predicted labels") 
print(model.predict(X))

print("true labels")
print(Y)

print("training accuracy") 
print(model.score(X, Y))


'''
bivariate case (X1, X2)
X1,X2 ~ N(0,1) 
in the first two cases X1 = X2 
in the latter three X1 indep X2 
polynomial kernel with degree 5 should work on the training set
'''
print("Bivariate (X1,X2), Standard Gaussian, label = 1 is X1=X2") 
Y = np.array([1,1,2,2,2], dtype=np.int32)
X = [0,0,0,0,0] 
X[0] = [np.random.normal(0,1,100), np.random.normal(0,1,100)] 
X[0][1] = X[0][0]  
X[1] = [np.random.normal(0,1,100), np.random.normal(0,1,100)] 
X[1][1] = X[1][0]  
X[2] = [np.random.normal(0,1,100), np.random.normal(0,1,100)] 
X[3] = [np.random.normal(0,1,100), np.random.normal(0,1,100)] 
X[4] = [np.random.normal(0,1,100), np.random.normal(0,1,100)] 
X = np.array(X, dtype = object)



print("smm with NuSVC and polynomial(10) kernel")
model = smm(kernel = lambda a,b: polynomial_kernel(a, b, 10),  
        base_svm = svm.NuSVC,  
        nu = 0.5,
        batch = 3,
        normalize = True)
model.fit(x = X, y = Y, verbose = True)

print("predicted labels") 
print(model.predict(X))

print("true labels")
print(Y)

print("training accuracy") 
print(model.score(X, Y))

'''
Causal pairs (X1,X2)
here labels are the causal direction 
'''
print("Causal pairs from gp_add")
gen = data.CausalPairGenerator("gp_add")
d, labels = gen.generate(100, npoints = 100)
X = d.to_numpy()
Y = labels.to_numpy()[:,0]
print("smm with SVR and  rbf(100) kernel")
model = smm(kernel = lambda a,b: rbf_kernel(a, b, 100), 
        base_svm = svm.SVC,
        batch = 4, 
        normalize = True)
model.fit(x = X, y = Y)

print("training score") 
print(model.score(X, Y))
gen = data.CausalPairGenerator("gp_mix")

d, labels = gen.generate(100, npoints = 100)
Xtest = d.to_numpy()
Ytest = labels.to_numpy()[:,0]

print("test score") 
print(model.score(Xtest, Ytest))

