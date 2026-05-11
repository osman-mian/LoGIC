from globe.TwoWayPriorityQueue import TwoWayPriorityQueue
from globe.graph import GraphUtil
from globe.TreeScore import TreeScore
from globe.DiscreteScore import DiscreteScore
import numpy as np
import itertools
import os
import concurrent.futures
import math
from typing import Any, Callable, Dict, List


def compute_phi(score, data, x1, x2):
    # This is executed during the initialization phase
    #print(x1, " <---> ",x2)
    if x1==x2: return x1,x2,0
    lim=20
    s_12 = score.compute_gain(data, x2, [], x1)
    s_21 = score.compute_gain(data, x1, [], x2)
    phi = s_12 - s_21  # NOTE: Gain is already multiplied by -1 inside compute_gain
    #print("------")
    #print("Phi ",x1,"/",x2,": ",phi)
    #if x1<lim and x2<lim and phi != 0 : print(f"Between {x1} and {x2} we have s_12:{np.round(s_12,2)} and s_21:{np.round(s_21,2)} with phi:{np.round(phi,2)}")
    return x1, x2, phi


class GreedyDagSearch:
    def __init__(self, **kwargs):
        sc_type = kwargs.get('score_func',None)
        use_reg = kwargs.get('regularize',False)
        alpha   = kwargs.get('alpha',None)
        pq      = kwargs.get('pq',None)
        score_dict={'ACID':DiscreteScore(use_reg), 'GLOBE':TreeScore()} 
        score = score_dict.get(sc_type,None)
        
        self.score = score if score is not None else TreeScore()
        self.tol = alpha if alpha is not None else 10
        self.pq = pq if pq is not None else TwoWayPriorityQueue({})  # for storing candidates
        self.gu = GraphUtil()
        workers = math.floor((os.cpu_count() - 3)/2)
        #print(f"Available CPUs: {os.cpu_count()}")
        #print(f"Using CPUs: {workers}")

        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        return

    def init(self, data: np.ndarray,prior_edges=None):
        #import ipdb;ipdb.set_trace()
        self.dt = data
        self.n = data.shape[0]
        self.dims = data.shape[1]
        self.network = np.eye(self.dims) * 0
        self.temp_g = np.eye(self.dims) * 0
        
        if prior_edges is None: prior_edges = np.ones((self.dims,self.dims))

        args = [
            (self.score, data, x1, x2)
            for x1 in range(self.dims)
            for x2 in range(x1, self.dims)
            if prior_edges[x1,x2]!=0 or prior_edges[x2,x1]!=0
        ]
        futures = {self.executor.submit(compute_phi, *arg): arg for arg in args}
        concurrent.futures.wait(futures)

        for future in futures:
            result = future.result()
            x1, x2, phi = result
            if not (np.abs(phi) >= self.tol):
                continue
            
            #if prior_edges is not None and prior_edges[x1,x2]!=0 or prior_edges[x2,x1]!=0:
            self.pq[(x1, x2)] = phi
            self.pq[(x2, x1)] = -phi
            self.temp_g[x1, x2] = 1
            self.temp_g[x2, x1] = 1

        #import ipdb;ipdb.set_trace()
        return

    def learn(
        self,
        data: np.ndarray,
        headers: List[str]=None,
        prior_edges = None
    ):
        self.init(data,prior_edges)
        if headers is None:
            headers = [str(i) for i in range(data.shape[1])]
            
        #return np.eye(data.shape[1])*0
        while True:
            #print("Starting forward phase")
            converge = self.forward(headers)
            if converge:
                break
            #print("Starting backward phase")
            converge = self.backward(headers)
            if converge:
                break

        self.score.reset();
        return np.vstack(self.network).astype(int)

    def backward(self,headers):
        converged = True

        for child in range(self.dims):
            edge_removed = True
            while edge_removed:
                edge_removed = False
                curr_parents = np.nonzero(self.network[:, child])[0].tolist()
                if len(curr_parents) <= 1:
                    continue

                s_full = self.score.compute(self.dt, child, curr_parents)
                set_size = len(curr_parents) - 1
                sets = itertools.combinations(curr_parents, set_size)

                # score candidates in parallel
                args = [(self.dt, child, list(set_)) for set_ in sets]
                futures = {
                    self.executor.submit(self.score.compute, *arg): arg for arg in args
                }
                concurrent.futures.wait(futures)

                for future in futures:
                    s_trunc = future.result()
                    arg = futures[future]
                    set_ = arg[-1]
                    if (s_trunc + self.tol) < s_full:
                        edge_removed = True
                        s_full = s_trunc
                        best_s = set_

                if edge_removed:
                    master_set = set(curr_parents)
                    sub_set = set(best_s)
                    removed_index = list(master_set - sub_set)[0]
                    #print("removed ",headers[removed_index]," -> ",headers[child])
                    self.network[removed_index, child] = 0
                    converged = False

        return converged

    def forward(self,headers):
        adds = 0
        pq = self.pq
        data = self.dt
        converged = True

        while pq.hasMore():
            # extract from queue and decipher
            k = pq.next_best()
            k_prime = (k[1], k[0])
            v = pq[k]
            parent = k[0]
            child = k[1]
            # remove forward AND backward directions from consideration
            pq.removeEntry(k)
            if not k == k_prime:
                pq.removeEntry(k_prime)

            self.temp_g[parent, child] = 0
            self.temp_g[child, parent] = 0
            # check edge validity
            flag1 = self.gu.CausesCycle(self.network, child, parent)
            flag2 = v > (-self.tol)

            if flag1 or flag2:
                continue

            # add to graph
            self.network[parent, child] = 1
            #print("adding ",headers[parent]," -> ",headers[child])
            converged = False
            adds = adds + 1
            cand_parents = np.nonzero(self.temp_g[:, child])[0].tolist()
            curr_parents = np.nonzero(self.network[:, child])[0].tolist()

            # score candidates in parallel
            args = [(data, child, curr_parents, op) for op in cand_parents]
            futures = {
                self.executor.submit(self.score_candidate, *arg): arg for arg in args
            }
            concurrent.futures.wait(futures)

        #print("Adds: ",adds)
        return converged

    def score_candidate(self, data, child, curr_parents, op):
        # This is executed during the forward phase
        pa_pa = np.nonzero(self.network[:, op])[0].tolist()
        ch_pa = curr_parents
        s_12 = self.score.compute_gain(data, child, ch_pa, op)
        s_21 = self.score.compute_gain(data, op, pa_pa, child)
        phi = s_12 - s_21
        self.pq[(op, child)] = phi
        self.pq[(child, op)] = -phi
        
        
    def compute_confidence(self,data,curr_parents,child):
        s1 = self.score.compute(data, child, [],cache=False)
        s2 = self.score.compute(data, child, curr_parents,cache=False)
        #print("Alone: ",s1)
        #print("With parents ",curr_parents,": ",s2)
        
        numerator = s1-s2
        denominator = s1 * 1.0
        confidence = numerator / denominator
        return confidence
