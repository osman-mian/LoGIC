from numpy import ndarray
from typing import Any, Callable, Dict, List
import numpy as np
import gc

from sklearn.tree import DecisionTreeRegressor
from mlinsights.mlmodel import PiecewiseRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from utils import save_fig_tup

from logic.base_regressors import MeanRegressor, DTRegressor

    
class PolyScore:
    def __init__(self,fl=False,cache_result=True):
        self.score_cache = {}
        self.model_cache = {}
        self.resolution_cache = None
        self.flag =fl
        self.use_cache=cache_result
        self.name_ ='Poly Score'
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


    def local_score_pwreg(self,X: ndarray, i: int, structure: List[int], parameters=None,debug=False) -> float:
        pa_count = len(structure) if len(structure)>0 else 1
        n,d		 = X.shape
        model		= 0
        if self.resolution_cache[i] <=0.001: self.resolution_cache[i]=0.001
        X = X[~np.isnan(X).any(axis=1)]
        if len(structure) == 0:
            residual = np.sum((X[:, i]-np.mean(X[:,i])) ** 2)
            sigmasq = (1.0*residual) / n;
            dgm	  = (n / (2 * np.log(2)) ) +  (n * 0.5 * np.log2(2 * np.pi * sigmasq ) ) +self.logN(np.mean(X[:,i])) - n*np.log2(self.resolution_cache[i])

            wrapped_regressor = MeanRegressor(np.mean(X[:,i]))
            bic = model+dgm

        else:
            bic = 9e99
            wrapped_regressor = None
            max_deg = int(np.floor(12 / len(structure)))
            for deg in range(1,max_deg):
                regressor  = Pipeline(steps=[('preprocessor', PolynomialFeatures(degree=deg, include_bias=True)),('estimator', LinearRegression())])
                regressor.fit(X[:,structure], X[:,i])
                predictions = regressor.predict(X[:,structure])
                temp_regressor = DTRegressor(regressor)

                #model score
                #print(regressor['estimator'].coef_)
                model = self.model_score(regressor['estimator'].coef_)
                
                #data given model
                residual    = np.sum((X[:, i]-predictions) ** 2)+0.0001
                sigmasq     = (1.0*residual) / n;
                dgm         = (n / (2.0 * np.log(2))) + (n * 0.5 * np.log2(2 * np.pi * sigmasq )) - n*np.log2(self.resolution_cache[i])
                
                temp_bic = model + dgm
                
                if debug:
                    tup1 = [(X[:,structure],predictions,'r'),(X[:,structure],X[:,i],'y')];tup1.reverse()
                    save_fig_tup(tup1,"./verify/"+str(structure)+"_"+str(i)+"_"+str(deg)+".png")
                    print(deg,": ",model," + ", dgm ," =  ",temp_bic," <? ",bic)
                    
                if deg==1 or temp_bic < bic:
                    bic = temp_bic
                    wrapped_regressor=DTRegressor(regressor)


        
        return bic,wrapped_regressor

    def compute(self,data, i: int, PAi: List[int],debug=False) -> float:
        if not self.use_cache: self.reset()

        #if self.resolution_cache is None:
        self.compute_resolution(data)

        if i not in self.score_cache:
            self.score_cache[i] = {}
            self.model_cache[i] = {}

        hash_key = tuple(sorted(PAi))

        if not self.score_cache[i].__contains__(hash_key):
            self.score_cache[i][hash_key],self.model_cache[i][hash_key] = self.local_score_pwreg(data, i, PAi,None,debug)
        #import ipdb;ipdb.set_trace()
        return self.score_cache[i][hash_key],self.model_cache[i][hash_key]


    def compute_normalized(self,data, i: int, PAi: List[int]) -> float:
        score, model = self.compute(data,i,PAi)
        score = score / data.shape[0]
        return score, model


    def compute_gain(self,data, i: int, PAi: List[int], j: int) -> float:
        #import ipdb;ipdb.set_trace()
        s1,_ = self.compute(data,i,PAi)
        PAi.append(j)
        s2,_ = self.compute(data,i,PAi)
        delta = s1 - s2
        #import ipdb;ipdb.set_trace()
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



