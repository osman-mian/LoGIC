from utils import *

import sys
import os
import traceback
import gc
from logic.logic import LOGIC
from logic.perfect_oracle import PerfectOracle
from logic.mdl_oracle import MDLOracle
from logic.globe_oracle import GLOBEOracle
from logic.poly_oracle import PolyOracle


from logic.mdl_score import MDLScore
from logic.poly_score import PolyScore
from logic.globe_score import GLOBEScore

from experiment_factory import ExperimentFactory
import networkx as nx

from acyclic_graph_generator import AcyclicGraphGenerator
from experiment_handler import Experiment
from common_structs import *
import datetime

np.random.seed(37)
import os

def gen_data(nodes=10,mech='gaussian_add',noise=0.4,idx=0,rcount=4000,pre_load=False,struc_fun=None,viz=False):
    foldname = './data/'+mech+'/'+str(nodes)+"_"+str(noise)+"/"
    fname = foldname + 'exp' + str(idx) + '.txt'
    gname = foldname + 'exp' + str(idx) + '_truth.txt'
    vname = './viz/temp/'+mech+'/'+str(nodes)+"_"+str(noise)+"/"+ 'exp' + str(idx) + '/'
    
    if pre_load:
        data            = load_data(fname)
        data            = standardize(data)
        graph           = load_graph(gname)
        dims            = graph.shape[1]
        
    else:
        
        
        if struc_fun is None:
            generator = AcyclicGraphGenerator(mech, npoints=rcount,nodes=nodes,initial_variable_generator=gaussian_cause,noise_coeff=noise)
            dt , G  = generator.generate() 
        else:
            #print("Custom...")
            pd=struc_fun(nodes)
            generator = AcyclicGraphGenerator(mech, npoints=rcount,nodes=pd.shape[1],initial_variable_generator=gaussian_cause,noise_coeff=noise)
            dt , G  = generator.generate(pre_dag=pd)
            
        graph = nx.to_numpy_array(G)
        data  = dt.to_numpy().reshape((-1,nodes))
            
        dims  = graph.shape[1]
        
        if not os.path.exists(foldname): os.makedirs(foldname) 
        np.savetxt(fname,  data, delimiter=",")
        np.savetxt(gname, graph, delimiter=",")
        data = standardize(data)
    
    if False or viz:
        if not os.path.exists(vname): os.makedirs(vname) 

        for i in range(dims):
                for j in range(dims):
                    if True or graph[i,j]!=0: 
                    #    print(i," -> ", j)
                        save_fig(data[:,i],data[:,j],vname+str(i)+'_'+str(j)+'.png',)
    
    return fname,data,graph

def main():
    rcount          = 2000 #if len(sys.argv)<2 else int(sys.argv[1])
    struct_id       = 0    #if len(sys.argv)<3 else int(sys.argv[2])
    mechs,noises,nodes,mask_probs,file_count,exp_fname = ExperimentFactory.toy_setup(rcount)
    tot_experiments = len(mechs) * len(noises) * len(nodes) * file_count  * len(mask_probs) 
    exp_counter     = 1
    
    structs = [None,chain,tree,collider,diamond,river,full]
    struct_fun = structs[struct_id]
    
    if int(sys.argv[1])==0:
        print("No priors")
    elif int(sys.argv[1])==1:
        print("Orc")
    else:
        print("Orc + gt");

    
    for node in nodes:
        for mech in mechs:
            acc_shd = 0.1
            acc_sid = 0.1
            for noise in noises:
                for idx in range(1,file_count+1):
                    gc.collect()
                    fname,dt,graph  = gen_data(node,mech,noise,idx,rcount,pre_load=False,struc_fun=struct_fun)
                    #print("(",datetime.datetime.now(),") ",exp_counter,"/",tot_experiments,": ",fname,": ",dt.shape,"/",np.sum(graph))
                    #exp_counter+=1
                    #continue
                    for p in mask_probs:
                        mask            = binary_sampler(1 - p, dt.shape[0], dt.shape[1])
                        xmis            = dt.copy()
                        xmis[mask == 0] = np.nan
                        dims = graph.shape[1]
                        try:
                            orc1 = GLOBEOracle()
                            orc2 = PerfectOracle(graph)
                            gt   = graph
                            if int(sys.argv[1])==0:
                                alg             = LOGIC(orc=orc1,sc=GLOBEScore(cache_result=False))
                                gt              = None
                                #print("No priors")
                            elif int(sys.argv[1])==1:
                                alg             = LOGIC(orc=orc2,sc=GLOBEScore(cache_result=False))
                                gt              = None
                                #print("Orc")
                            else:
                                alg             = LOGIC(orc=orc2,sc=GLOBEScore(cache_result=False))
                                #print("Orc+gt")
                            midata,pred   = alg.fit_transform(np.copy(xmis),gt)
                            shd,sid,paid,tp,fp,fn,ac = calc_stats(pred,graph)
                            acc_shd+=shd
                            acc_sid+=sid
                            #import ipdb;ipdb.set_trace()
                            rmse,ir            = rmse_loss(dt,midata,mask)
                            print(exp_counter,"/",tot_experiments,", ",fname,", Oracle: ",alg.oracle.name_,", prob: ",p,", Tot: ",np.sum(graph),", SHD: ",shd,", SID: ",sid,", PAID: ",paid,", TP: ",tp,", FP: ",fp,", FN: ",fn,", RMSE: ",rmse,", IR: ",ir)
                            #print("----")
                        except Exception as e:
                            print(exp_counter,"/",tot_experiments,", ",fname,", Oracle: ",alg.oracle.name_," failed")
                            print(e)
                            #traceback.print_exc()
                            #import ipdb;ipdb.set_trace()
                        gc.collect()
                        exp_counter+=1
            acc_shd-=0.1
            acc_sid-=0.1
            print("Avg SHD: ",np.round(acc_shd/(len(mask_probs)*len(noises)*file_count),2)) 
            print("Avg SID: ",np.round(acc_sid/(len(mask_probs)*len(noises)*file_count),2)) 
            
            
            print('----------')
                        #print('---')
          
    print("#############")
        
    
main();