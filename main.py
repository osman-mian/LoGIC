import sys
import os
import gc #garbage collector
from datetime import datetime

from methods import LOGICX

from utils import * #includes data loader and statistic calculators
np.random.seed(37)

    
def main():
    
    #Load the Data
    data  = standardize(load_data("./data/realworld/reged5a.txt"))[0:5000,:] #total length is 20k rows, we just take 5k for this example.
    graph = load_graph("./data/realworld/reged5a_truth.txt")    #must be a binary comma-separated adjacency matrix
    
    #Initialize Components of LOGIC
    withOracle      = True
    withoutOracle   = False
    algos           = [LOGICX(withOracle),LOGICX(withoutOracle)]

    #Create Missing Data from complete dataset
    p               = 0.10        #missing probability
    dims            = data.shape[1]
    mask            = binary_sampler(1 - p, data.shape[0], data.shape[1])
    xmis            = data.copy()
    xmis[mask == 0] = np.nan
    mrows_ids          = np.sum(mask,axis=1)<data.shape[1]

    #run the algorithms
    for alg in algos:
        try:
            ts                     = datetime.now().strftime("%d.%m.%y:%H:%M")
            print("[",ts,"] BEGIN: ",alg.name_)

            midata,pred,iscpdag    = alg.run(np.copy(xmis),mrows_ids,graph)

            ts                     = datetime.now().strftime("%d:%m:%y:%H:%M")
            print("[",ts,"] END: ",alg.name_)

            print("Predicted graph: ")
            print(pred)
            
            if graph is not None:
                rmse,ir               = rmse_loss(data,midata,mask)
                shd,sid,paid,tp,fp,fn = calc_ud_stats(pred,graph,iscpdag)
                print("Alg: ",alg.name_,", prob: ",p,", Tot: ",np.sum(graph),", nSHD: ",np.round(shd*1.0/(dims * (dims-1)),2),", nSID: ",np.round(sid*1.0/(dims * (dims-1)),2),", TP: ",tp,", FP: ",fp,", FN: ",fn,", RMSE: ",rmse,", Imputation Rate: ",ir)

        except Exception as e:
            print(alg.name_," failed")
            print(e)
            
            
        gc.collect()
        
    
main();