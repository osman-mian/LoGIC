import numpy as np
from logic.node import Node
from logic.mdl_oracle import MDLOracle
from logic.mdl_score import MDLScore
from globe.gds import GreedyDagSearch


from logic.base_regressors import MeanRegressor, DTRegressor

import itertools
import gc
from ordered_set import OrderedSet
from utils import save_fig_tup
import concurrent.futures
import time
class LOGIC:
    
    def __init__(self,orc=None,sc=None):
        self.is_init = True
        self.oracle = orc if orc is not None else MDLOracle()
        self.score  = sc  if sc  is not None else MDLScore(cache_result=False)
        
        #to be used later in calculation
        self.pairwise=None

    def fit_transform(self,data,gt=None):
        dims  = data.shape[1]
        rows  = data.shape[0]
        graph = np.eye(dims)*0
        clusters     = self.determine_clusters(data)
        source_list  = self.find_sources(data,clusters)
        idata,igraph = self.learn_impute(data.copy(),source_list)
        return idata, igraph
    
    def cluster_transform(self,data,gt=None):
        dims  = data.shape[1]
        rows  = data.shape[0]
        graph = np.eye(dims)*0
        tsources     = np.where(np.sum(gt,axis=0)==0)[0]
        clusters     = self.determine_clusters(data)
        source_list  = self.find_sources(data,clusters)
        return source_list
    
            
   
    def learn_impute(self,data,sources):
        dims      = data.shape[1]
        total     = set([i for i in range(dims)]) 
        remaining = set([i for i in range(dims)]) 
        graph     = np.eye(dims)*0
        S         = sources
        
        #print("Total: ",remaining)
        while len(S)!=0:
            #print("Considering the SET ",S," in this round...")
            remaining     = remaining - set(S)
            #print("Remaining: ",remaining)
            S_next        = []
            S_descendents = self.find_next_descendents(S,remaining,data)     #choose all nodes such that one of the S is its best "parent"
            
            #print("Possible descendents of ",S," are: ",S_descendents)
            for V in S_descendents:
                #print("---\n Assessing ",V)
                V_desc          = self.choose_best_descendent(V,data,remaining)   #choose a descendent of current variable from remaining variables. This descendent will be used to capture noise term
                #print("---\n",V," has ",V_desc," as its best descendent")
                #print("Learn ",V," by using ",total - remaining," -> ",V, " -> ",V_desc)
                data[:,V], pa_v = self.impute(V,total-remaining,V_desc,data) #impute using parents and descendent for noise. Choose parents as the ones that minimize a given score
                #print("I found ",pa_v," as the final parents")
                graph[pa_v,V]   =1 #add edges to graph

                if V_desc is not None:
                    #print("Because ",V," had a descendent ",V_desc," we will add it to the next round")
                    S_next.append(V)                                         #if this node has a descendent, that will also have to be imputed, else this was the sink node and nothing further needs to be done
                
            S = S_next.copy()                                                #repeat the cycle for variables that still have descendents

        #if len(remaining)>0:
        #    print("oops, we never reached: ",remaining)    
            
        return data,graph
    
    def impute(self, target, candidates, child, dt):
        data = np.copy(dt)
        #these will be used for imputation
        source_vars = list(candidates).copy()
        if False and child is not None:
            source_vars.append(child)
        
        #need this to get non-missing rows cleanly
        all_vars = source_vars.copy()
        all_vars.append(target)


        #impute
        local_index     = [i for i in range(len(source_vars))]
        chunk           = np.copy(self.get_non_missing_rows(data,all_vars))              #get non-missing rows
        target_index    = len(all_vars) - 1                                              #we always put the variable under consideration at the end
        predictor       = self.score.learn(chunk[:,local_index],chunk[:,target_index])   #learn a model using possible parents and the child
        
        #find indices where this target is missing but all the predictors are present
        target_missing_idx = np.isnan(data[:,target])

        #find where descendent is missing and set it to zero for a higher imputation rate
        if child is not None:
            desc_missing_idx= np.isnan(data[:,child])
            data[desc_missing_idx,child] = 0
        
        
        source_present_idx = ~np.isnan(data[:,source_vars]).any(axis=1)
        imputable_index    = np.where( target_missing_idx & source_present_idx)[0]
        if imputable_index.shape[0]>0:
            data[imputable_index,target] = predictor.predict(data[np.ix_(imputable_index,source_vars)]) 
        predicted_target             = data[:,target]
                                                  
        chunk           = np.copy(self.get_non_missing_rows(data,all_vars)) 
        num_parents = len(source_vars) if child is None else len(source_vars)-1          #We consider upto len(source_vars) - 1 because we dont want the descendent to be considered as a parent
        local_source    = [i for i in range(num_parents)]                               
        local_pa_v      = self.find_parents(target_index,local_source,chunk)             #Find the parents' local indices in the extracted chunk
        pa_v            = [ source_vars[i] for i in local_pa_v]                          #Convert them back to original indices
        return predicted_target, pa_v
        
        
    
    def choose_best_descendent(self,V,data,candidates):
        dims      = data.shape[1]
        pairwise  = np.zeros(dims)
        desc_list = []
        desc      = None
        
        
        #here we compute best set of parents for each of the remaining candidate
        for remaining in candidates:
            best_phi=-9e99
            best_pa = -1
            for i in range(dims):
                if i==remaining: continue
                
                trunc_mod_data    = self.get_non_missing_rows(data,[i,remaining])
                g1                = self.score.compute_gain(trunc_mod_data,1,[],0); #i -> remaining
                g2                = self.score.compute_gain(trunc_mod_data,0,[],1); #remaining -> i
                phi               = g1 - g2
                
                
                if phi > best_phi:
                    best_phi = phi
                    best_pa  = i
                    
            #check if the best parent is one of the current source variables
            if best_pa == V:
                desc_list.append((best_phi,remaining))
        
        #If multiple descendents found, choose that which is the strongest
        if len(desc_list)>0:
            best_did=0
            best_phi=desc_list[best_did][0]

            for did in range(1,len(desc_list)):
                if desc_list[did][0] > best_phi:
                    best_did = did
                    best_phi = desc_list[did][0]

            desc = desc_list[best_did][1]
            
        return desc
            
        
    
    def find_next_descendents(self,sources,candidates,data):
        dims      = data.shape[1]
        pairwise  = np.zeros(dims)
        desc_list = []
        
        
        #here we compute best set of parents for each of the remaining candidate
        for remaining in candidates:
            best_phi= -9e99
            best_pa = -1
            for i in range(dims):
                if i==remaining: continue
                
                trunc_mod_data    = self.get_non_missing_rows(data,[i,remaining] ) #get non missing rows
                g1                = self.score.compute_gain(trunc_mod_data,1,[],0) #parent    -> remaining
                g2                = self.score.compute_gain(trunc_mod_data,0,[],1) #remaining -> parent
                phi               = g1 - g2
                #print(i," -> ", remaining," = ",phi)
                #positive phi means a parent has a stronger direction than the child (remaining) variable
                if phi > best_phi:
                    best_phi = phi
                    best_pa  = i
                    
            #check if the best parent is one of the current source variables
            if best_pa in sources:
                desc_list.append(remaining)      #if yes, this is a possible descendent of one of the sources
                
                
        return desc_list
    
    def find_sources(self,data,clusters):
        source_candidates = []
        source_list       = []#{}
        source_count      = len(clusters)
        
        for i in range(source_count):
            ci = set(clusters[i])
            for j in range(source_count):
                if i==j: continue
                cj = set(clusters[j])
                ci = ci - cj
                
            source_candidates.append(list(ci))
            

        #find actual source from shortlisted
        for candidate_set in source_candidates:
            if len(candidate_set)==0: continue
            source = self.get_source(candidate_set,data)
            source_list.append(source)
            #print("choose ",source, " from ", candidate_set)
        return source_list
    
    def get_source(self,candidates,data):
        dims = len(candidates)
        if dims==1: return candidates[0]
        accumulator = np.zeros(dims)
        acc_bits    = np.zeros(dims)
        for i in range(dims):
            c1 = candidates[i]
            for j in range(i+1,dims):
                
                c2 = candidates[j]
                trunc_mod_data    = self.get_non_missing_rows(data,[c1,c2])
                
                g1         = self.score.compute_gain(trunc_mod_data,1,[],0); #c1 -> c2
                g2         = self.score.compute_gain(trunc_mod_data,0,[],1); #c2 -> c1
                
                phi = g1 - g2
                
                if phi   > 0 : #meaning g1 had a bigger gain
                    accumulator[i]+=1
                elif phi < 0 : #meaning g2 had a bigger gain
                    accumulator[j]+=1

                acc_bits[i] +=np.round(phi,3)
                acc_bits[j] -=np.round(phi,3)
 
        #get the one with highest win rate
        flag=True
        while flag:
            flag=False
            #try:
            max_val = accumulator.max()
            #except:
            #    import ipdb;ipdb.set_trace()
            s_id = np.where(accumulator == max_val)[0]     #find maximum winners
            if max_val!=1:                                 #only run this loop if there can be a chance to continue in case of failure (not the best termination but shouldn't get this far in practice tbh)
                rem_set=[]                 
                for sid in s_id:
                    if acc_bits[sid]<0:                    #if a winner has overall negative gain, mark it for removal
                        rem_set.append(sid)
                s_id = list(set(s_id) - set(rem_set))      #remove the negatives
                if len(s_id)==0:                           #if everything was negative, try with a new set
                    flag=True
                    max_val-=1
                
            
        
        #if there is only one
        if len(s_id)==1:
            src = candidates[s_id[0]]
        else: #tie break using bit gain
            best_gain = -9e99
            source    = -1
            for sid in s_id:
                tied_source   = sid
                
                if acc_bits[tied_source] > best_gain:
                    best_gain = acc_bits[tied_source]
                    source    = tied_source
            src = candidates[source]
                
        cdd = [candidates[sid] for sid in s_id]
        return src
                
        
    def determine_clusters(self,data,seed_node=0):
        dims         = data.shape[1]
        node_array   = []
        cluster_list = [] #this will contain a list of connected components of a graph, ideally one per source node.
        self.oracle.init_independence(dims)

        init_cluster = [i for i in range(dims)]        # define first cluster to contain all variables
        #print(seed_node)
        first_cluster,alt_cluster =self.split_cluster(init_cluster,data,seed_node)
        #print(seed_node,": ",first_cluster)
        #import ipdb;ipdb.set_trace()
        split_cluster, extra_nodes = self.cluster_correction(first_cluster,data) # if we accidentally started at a collider or its child, init cluster will contain two sources. We need to disentangle them.
        cluster_list.append(split_cluster)          
        alt_cluster.extend(extra_nodes)
        #if len(alt_cluster)>0: print("After correction: ",split_cluster," / ",alt_cluster)
        
        finished = len(alt_cluster)==0              #nothing left to do if there are not variables left in the alternate cluster

        while not finished:
            #print("Splitting: ",alt_cluster)
            cluster, _ = self.split_cluster([i for i in range(dims)],data,alt_cluster[0])
            #print(alt_cluster[0],": ",cluster)
            cluster_list.append(cluster)
            
            s1 = set([i for i in range(dims)])
            s2 = set()
            
            for t_cluster in cluster_list:
                s2 = s2.union(set(t_cluster))
            
            finished = s2==s1
            alt_cluster = list(s1 - s2)
            #print("New: ",cluster," / ",alt_cluster)
        return cluster_list
    
    def cluster_correction(self,cluster,data):
        dims = data.shape[1]
        for i in cluster:
            for j in cluster:

                if i!=j and self.oracle.is_independent(i,j,data):
                    #print("Found: ",i," indep. ", j)
                    s1,s2   = i,j            #we found two nodes that are independent, should search for connected components from these
                    c1_full = cluster.copy()
                    c1_full.remove(s2)
                    
                    c2_full = cluster.copy()
                    c2_full.remove(s1)
                    cluster1,_ =self.split_cluster(c1_full,data,c1_full.index(s1))
                    cluster2,_ =self.split_cluster(c2_full,data,c2_full.index(s2))
                    #print(s1,":sp: ",cluster1)
                    #print(s2,":sp: ",cluster2)
                    return [cluster1,cluster2]
                
        return [cluster,[]]
        
    def split_cluster(self,cluster,data,seed_idx=0):
        init_cluster = cluster.copy()
        alt_cluster  = []
        dims = data.shape[1]
        seed_node=cluster[seed_idx]
        
        #print("Splitting on: ",seed_node)
        for i in cluster:
            if i==seed_node: continue
            #if a variable is independent of seed node, put it in a different cluster
            if self.oracle.is_independent(seed_node,i,data):
                alt_cluster.append(i)
                init_cluster.remove(i)

        return init_cluster, alt_cluster
    
    def find_parents(self,child,poss_pa,data):
        #print("Choosing for ",child)
        converged = True
        dims = data.shape[1]
        edge_removed = True;

        #import ipdb;ipdb.set_trace()
        while edge_removed:
            edge_removed  = False;
            curr_parents  = poss_pa
            
            #find ids that are complete
            idxs             = poss_pa.copy()
            tpx=poss_pa.copy()
            idxs.append(child)
            mod_data         = self.get_non_missing_rows(data,idxs)
            mod_curr_parents = [pid for pid in range(len(curr_parents))]
            mod_child        = len(idxs)-1

            #print("Size: ",mod_data.shape)
            s_full,c_model    = self.score.compute_normalized(mod_data,mod_child,mod_curr_parents)
            #print(tpx," -> ",child," = " , s_full)

            set_size      = len(curr_parents) - 1
            if set_size < 0: break
            sets          = itertools.combinations(mod_curr_parents,set_size)
            
            for set_ in sets:
                trunc_idxs = [curr_parents[ss] for ss in list(set_)]           #for this set, get back the actual indexes
                trpx = trunc_idxs.copy()
                trunc_idxs.append(child)                                       #add child to the list
                trunc_mod_data = self.get_non_missing_rows(data,trunc_idxs)    #get data for which child and this truncated parent-set is not Nan.
                trunc_mod_curr_parents = [ti for ti in range(len(list(set_)))]
                trunc_mod_child = len(trunc_idxs)-1
                s_trunc,s_scm = self.score.compute_normalized(trunc_mod_data,trunc_mod_child,trunc_mod_curr_parents );
                #print(trpx," -> ",child," = " , s_trunc)
                
                if s_full > s_trunc-int(not edge_removed)*0.01: #only check for no-hypercompressibility if the full_set is still in its original form
                    edge_removed   = True
                    s_full         = s_trunc
                    best_s         = set_
                    best_scm       = s_scm
                    

            if edge_removed:
                master_set         = set(mod_curr_parents);
                sub_set            = set(best_s);
                removed_index      = list(master_set - sub_set)[0];
                poss_pa            = [curr_parents[ss] for ss in sub_set]
                converged          = False
                c_model            = best_scm

        #print(poss_pa," returned")
        #import ipdb;ipdb.set_trace()
        return poss_pa#,c_model
    
    
    def get_non_missing_rows(self,data,idxs):
        chunk            = data[:,idxs]
        mod_data         = chunk[~np.isnan(chunk).any(axis=1)]
        return mod_data
    
    