from numpy import ndarray
from typing import Any, Callable, Dict, List
import numpy as np
import gc

from sklearn.tree import DecisionTreeRegressor
from mlinsights.mlmodel import PiecewiseRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from logic.base_regressors import MeanRegressor, DTRegressor

    
class MDLScore:
    def __init__(self,fl=False,cache_result=True):
        self.score_cache = {}
        self.model_cache = {}
        self.resolution_cache = None
        self.flag =fl
        self.use_cache=cache_result
        self.name_ = 'MDL Score'
        #print("I will use the cache: ",self.use_cache)

    def logN(self,z):
        z = np.ceil(z);

        if z < 1 :
            return 0;
        else :
            log_star = np.log2(z);
            sum_ = log_star;

            while log_star > 0:
                log_star = np.log2(log_star);
                sum_ = sum_+log_star;

            return sum_ + np.log2(2.865064)


    def model_score(self,coeff):
        Nans = np.isnan(coeff);

        if any(Nans):
            print ('Warning: Found Nans in regression coefficients. Setting them to zero...')
        coeff[Nans]=0;
        sum_ =0;
        for c in coeff:
            if np.abs(c)>1e-12:
                c_abs =  np.abs(c);
                c_dummy = c_abs;
                precision = 1;

                while c_dummy<1000:
                    c_dummy *=10;
                    precision+=1;
                sum_ = sum_ + self.logN(c_dummy) + self.logN(precision) + 1
        return sum_;


    def local_score_pwreg(self,X: ndarray, i: int, structure: List[int], parameters=None) -> float:
        pa_count = len(structure) if len(structure)>0 else 1
        n,d		 = X.shape
        model		= 0
        if self.resolution_cache[i] <=0.001: self.resolution_cache[i]=0.001
        X = X[~np.isnan(X).any(axis=1)]
        if len(structure) == 0:
            residual = np.sum((X[:, i]-np.mean(X[:,i])) ** 2)
            sigmasq = (1.0*residual) / n;
            dgm	  = (n / (2 * np.log(2)) ) +  (n * 0.5 * np.log2(2 * np.pi * sigmasq ) ) +self.logN(np.mean(X[:,i])) - n*np.log2(self.resolution_cache[i])

            total_count = 1
            leaf_count  = 1
            inner_count = 0
            wrapped_regressor = MeanRegressor(np.mean(X[:,i]))


        else:
            max_port          = np.log2(n)
            sample_size       = int(n/max_port)
            regressor         = PiecewiseRegressor(verbose=False, binner=DecisionTreeRegressor(min_samples_leaf=sample_size))
            regressor.fit(X[:,structure], X[:,i])
            predictions       = regressor.predict(X[:,structure])
            wrapped_regressor = DTRegressor(regressor)
            
            #data given model
            residual          = np.sum((X[:, i]-predictions) ** 2)+0.0001
            sigmasq           = (1.0*residual) / n;
            dgm               = (n / (2.0 * np.log(2))) + (n * 0.5 * np.log2(2 * np.pi * sigmasq )) - n*np.log2(self.resolution_cache[i])

            #model
            total_count = regressor.binner_.tree_.node_count
            leaf_count  = regressor.binner_.tree_.n_leaves.item()
            inner_count = total_count - leaf_count

            for est in regressor.estimators_:
                model += self.model_score(est.coef_)

        model      += self.logN(total_count) + total_count
        model      += inner_count * (np.log2(pa_count) + np.log2(n) + 2*np.log2(total_count))		#for each inner node, identify the point to split on and the axis to split by, and which nodes to point to
        bic         = model + dgm
        if self.flag: import ipdb;ipdb.set_trace()
        return bic,wrapped_regressor

    def compute(self,data, i: int, PAi: List[int]) -> float:
        if not self.use_cache: self.reset()

        if self.resolution_cache is None:
            self.compute_resolution(data)

        if i not in self.score_cache:
            self.score_cache[i] = {}
            self.model_cache[i] = {}

        hash_key = tuple(sorted(PAi))

        if not self.score_cache[i].__contains__(hash_key):
            self.score_cache[i][hash_key],self.model_cache[i][hash_key] = self.local_score_pwreg(data, i, PAi)
        #import ipdb;ipdb.set_trace()
        return self.score_cache[i][hash_key],self.model_cache[i][hash_key]


    def compute_normalized(self,data, i: int, PAi: List[int]) -> float:
        score, model = self.compute(data,i,PAi)
        score = score / data.shape[0]
        return score, model


    def compute_gain(self,data, i: int, PAi: List[int], j: int) -> float:
        #import ipdb;ipdb.set_trace()
        import ipdb;ipdb.set_trace()

        s1,_ = self.compute(data,i,PAi)
        PAi.append(j)
        s2,_ = self.compute(data,i,PAi)
        delta = s1 - s2
        import ipdb;ipdb.set_trace()
        delta = delta.item() if delta > 0 else 0
        return delta/data.shape[1]



    def score_graph(self,data,graph):
        dims 	= graph.shape[1]
        self.compute_resolution(data)
        sc 		= 0
        for i in range(dims):
            pa_i = np.nonzero(graph[:,i])[0].tolist()
            sc  += self.compute(data,i,pa_i)

        self.resolution_cache=None
        return sc


    def reset(self):
        self.score_cache = {}
        self.model_cache = {}
        self.resolution_cache = None

    def compute_resolution(self,data):
        self.resolution_cache = {}
        rows = data.shape[0]
        for i in range(data.shape[1]):
            c1 = list(data[:,i])
            c1.sort()
            c2 = c1.copy()
            c3 = np.min(np.array(c2)[1:rows] - np.array(c1)[0:rows-1])
            c3 = 0.001 if c3 <= 0 else c3
            self.resolution_cache[i] = np.round(c3,5)

        #print(self.resolution_cache)
        #import ipdb;ipdb.set_trace()



