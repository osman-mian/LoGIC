from sep_distances.metrics import *
from sep_distances.mixed_graph import LabelledMixedGraph
import numpy as np


def calc_sep_dist_cpdag(pred,truth):
    
    pred  = to_mixed_graph(pred)
    truth = to_mixed_graph(truth)
    
    #import ipdb;ipdb.set_trace()
    
    return dist_cpdag(pred,truth)
        
    
def calc_sep_dist_dag(pred,truth):
    
    pred  = to_dag_graph(pred)
    truth = to_dag_graph(truth)
    
    if pred is None:
        #print("Potential cycle in graph..");
        return np.nan,np.nan
    #import ipdb;ipdb.set_trace()
    
    return dist_dag(pred,truth)

def dist_dag(pred,truth):
    try:
        
        P_AID = sym_parent_AID_DAGs(pred,truth)
        A_AID = sym_ancestor_AID_CPDAGs(pred,truth)

        return np.round(P_AID,2),np.round(A_AID,2)
    
    except Exception as e:
        print("Exception in calc dist: ")
        print(e)
        
        
    return np.nan,np.nan
    

def dist_cpdag(pred,truth):
    try:
        P_AID = sym_parent_AID_CPDAGs(pred,truth)
        A_AID = sym_ancestor_AID_CPDAGs(pred,truth)

        return np.round(P_AID,2),np.round(A_AID,2)
    
    except Exception as e:
        print("Exception in calc dist: ")
        print(e)
        
        
    return np.nan,np.nan
    
    
def to_mixed_graph(graph):
    d = graph.shape[1]
    nodes = [i for i in range(d)]

    G1 = LabelledMixedGraph()
    
    for n in nodes: G1.add_node(n)

    for i in range(d):
        for j in range(i+1,d):
            if graph[i,j]==1 and graph[j,i]==0:
                G1.add_directed(i, j)
            if graph[i,j]==0 and graph[j,i]==1:
                G1.add_directed(j, i)
            if graph[i,j]==1 and graph[j,i]==1:
                G1.add_undirected(i, j)

    return G1

def to_dag_graph(graph):
    d = graph.shape[1]
    nodes = [i for i in range(d)]

    G1 = LabelledMixedGraph()
    
    for n in nodes: G1.add_node(n)

    for i in range(d):
        for j in range(d):
            if graph[i,j]==1 and graph[j,i]==1:
                return None
            
            if graph[i,j]==1:
                G1.add_directed(i, j)

    return G1


'''sanity check
g = np.zeros((5,5))
g[0,1]=1
g[0,2]=1
g[1,3]=1
g[2,3]=1
g[3,4]=1

print(g)

pred = g.copy()
g[3,4]=0
truth = g.copy()

print(calc_sep_dist_dag(pred,truth))
#'''