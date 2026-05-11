from experiment_handler import Experiment
import sys
import gc
import datetime
import numpy as np
from utils import *
import pandas as pd


#Step 1: Load the desired pickle file after running your desired experiments by updating the fname variable.
#Step 2: Uncomment the query related to the experiment you ran.
#Step 3: Run the code to view the summarized results for that experiment.
def main():
    fname = sys.argv[1]# './res/tot_res_1000.pkl'
    experiment = Experiment([],fname,True)
    experiment.exp_list = experiment.exp_list.drop(columns=["predicted","graph"])
    
    experiment.exp_list["oracle"] = experiment.exp_list["oracle"].str.strip()

    #normalize SID and SHD
    #experiment.exp_list['nodes']= experiment.exp_list.apply(lambda row: row["graph"].shape[1], axis=1)
    nds = experiment.exp_list['nodes']
    experiment.exp_list['sid'] = experiment.exp_list['sid'] / (nds*nds - nds)
    experiment.exp_list['shd'] = experiment.exp_list['shd'] / (0.5*(nds*nds - nds))
    
    
    querys ={}
    #querys["apxt1"]   = "select count(oracle) as exp, oracle as method, nodes, round(avg(shd),2) as shd from results where rmse>-1            group by nodes, oracle order by nodes, oracle"
    #querys["apxt2"]   = "select count(oracle) as exp, oracle as method, prob , round(avg(shd),2) as shd from results where rmse>-1            group by prob , oracle order by prob , oracle"
    #querys["fig1"]   = "select count(oracle) as exp, oracle as method, nodes, round(avg(sid),2) as sid, round(avg(rmse),2) as rmse, round(avg(ir),2) as IR from results where rmse>-1            group by nodes, oracle order by nodes, oracle"
    #querys["fig2"]   = "select count(oracle) as exp, oracle as method, prob , round(avg(sid),2) as sid, round(avg(rmse),2) as rmse, round(avg(ir),2) as IR from results where rmse>-1            group by prob , oracle order by prob , oracle"
    #querys["fig3_1"] = "select count(oracle) as exp, oracle as method,        round(avg(sid),2) as sid, round(avg(rmse),2) as rmse, round(avg(ir),2) as IR from results where rmse>-1 and mnar=0 group by oracle order by oracle"
    #querys["fig3_2"] = "select count(oracle) as exp, oracle as method,        round(avg(sid),2) as sid, round(avg(rmse),2) as rmse, round(avg(ir),2) as IR from results where rmse>-1 and mnar=1 group by oracle order by oracle"
    #querys["fig3_3"] = "select count(oracle) as exp, oracle as method,        round(avg(sid),2) as sid, round(avg(rmse),2) as rmse, round(avg(ir),2) as IR from results where rmse>-1 and so=1   group by oracle order by oracle"
    #querys["fig3_4"] = "select count(oracle) as exp, oracle as method,        round(avg(sid),2) as sid, round(avg(rmse),2) as rmse, round(avg(ir),2) as IR from results where rmse>-1 and so=0   group by oracle order by oracle"
    #querys["tab1"]    = "select count(oracle) as exp, oracle as method, mnar, nodes, round(avg(sid),2) as sid, round(avg(rmse),2) as rmse, round(avg(ir),2) as IR from results where rmse>-1      group by nodes, mnar, oracle order by mnar,nodes,rmse"
    #querys["tab0a"]   = "select count(oracle) as exp, oracle as method, prob,  round(avg(F1),2) as F1 from results                         group by prob, oracle order by prob, oracle"
    #querys["tab0b"]   = "select count(oracle) as exp, oracle as method, nodes,  round(avg(F1),2) as F1 from results                         group by nodes, oracle order by nodes, oracle"
    
    #for missdag
    querys["apxt1"]   = "select count(oracle) as exp, oracle as method, nodes, round(avg(sid),2) as sid from results where rmse>=-1            group by nodes, oracle order by nodes, oracle"
    querys["apxt2"]   = "select count(oracle) as exp, oracle as method, prob , round(avg(sid),2) as sid from results where rmse>=-1            group by prob , oracle order by prob , oracle"

    
    for q in querys.keys():
        res = experiment.query(querys[q])
        print(res)
    return


main()


'''
    exp_headers     = ['fname','tot','samples','so','oracle','prob','shd','sid','paid','tp','fp','fn','rmse','ir','predicted','graph']

    #experiment2 = Experiment([],'./res/src_res_mnar_1000.pkl',True)
    #df_combined = pd.concat([experiment.exp_list, experiment2.exp_list], ignore_index=True)
    #experiment.exp_list = df_combined

    #decide what to generate results on
    #metrics = ['nodes','noise','bins','samples']
    #qtys =['paid']#,'paid','aaid']
    #save = False

    #q1 = "select count(oracle),"+typ+", oracle,nodes,round(avg(sid),2) as sid,round(avg(paid),2) as paid, round(avg(rmse),2) as rmse, round(avg(ir),2) from results where rmse>-1 group by nodes, "+typ+", oracle order by "+typ+" desc,nodes,rmse"
    #q2 = "select count(oracle),"+typ+", oracle,round(avg(sid),2) as sid,round(avg(paid),2) as paid, round(avg(rmse),2) as sse, round(avg(ir),2) from results where rmse>-1 group by "+typ+", oracle order by "+typ+" asc,paid"
    #q3 = "select oracle,avg(paid) from results where oracle LIKE '%MIRACLE%'"
   
    for metric in metrics:
        for qty in qtys:
            #define query
            select_part = "select "    + metric     + ",method, round(avg("+qty+"),2) as "+qty+" from results "
            group_part  = "group by method," + metric     + " "
            order_part  = "order by "        +metric      + "," + qty

            q = select_part + group_part + order_part         #combine
            print(select_part);
            res = experiment.query(q)                      #execute
            #result = res.pivot(index=metric, columns='method', values=qty)
            print(res)
            if save:
                result.to_csv("./res/"+metric+"_"+qty+".tab", sep='\t', index=True)
            gc.collect()
            print(datetime.datetime.now())
            print("---------")
        
'''