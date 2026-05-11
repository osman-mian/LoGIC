from numpy import ndarray
from typing import Any, Callable, Dict, List
import numpy as np
import gc

from sklearn.tree import DecisionTreeRegressor
from mlinsights.mlmodel import PiecewiseRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class MeanRegressor:
    def __init__(self,mean):
        self.mu=mean

    def predict(self,pas):
        return self.mu

class DTRegressor:
    def __init__(self,reg):
        self.regressor = reg

    def predict(self,data):
        preds = data[:,0]*0
        filter_row  = ~np.isnan(data).any(axis=1)
        nan_rows    =  np.isnan(data).any(axis=1)

        preds[nan_rows]   = np.nan
        preds[filter_row] = self.regressor.predict(data[filter_row,:])

        return preds.reshape((-1,1))
    
class MDLScore:
    def __init__(self,graph,fl=False,cache_result=True):
        self.graph = graph
        self.score_cache = {}
        self.model_cache = {}
        self.resolution_cache = None
        self.flag =fl
        self.use_cache=cache_result
        self.name_ = 'MDL Score'
        self.order = self.find_layer_order()
        
    def find_layer_order(self):
        graph = self.graph
        queue = []
        visited = [False for i in range(graph.shape[1])]
        layer = 1
        sources = np.argwhere(np.sum(graph,axis=0)==0)

        #all sources go in by default as layer 1
        for src in sources:
            queue.append( (src.item(),layer) )

        #print(queue)
        while len(queue)!=0:
            node, layer = queue.pop(0)
            visited[node] = layer
            children = np.argwhere(graph[node,:]!=0)
            for c in children: queue.append((c.item(),layer+1))

        if False in visited: print("Error in traversal, atleast one node was left unvisited")
        
        return visited


    def compute(self,data, i: int, PAi: List[int]) -> float:
        
        return self.visited[i]


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



