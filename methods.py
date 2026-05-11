import numpy as np
import concurrent.futures
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

# I have implemented LogicX this way so that it is easy to extend for a competitor method. 
# Simply Implement your method like I have implemented LogicX.
# Then you can use the main.py file as-is by simply initializing your method in algos list and you are essentially done.

def _timed_run_wrapper(queue, instance, data, mrows=None, tgraph=None):
    try:
        result = instance.run(data, mrows, tgraph)
        queue.put(("ok", result))
    except Exception:
        tb = traceback.format_exc()
        queue.put(("err", tb))
        
class BaseMethod:
    #I wrote the ProcessPoolExecutorCode, ChatGPT converted it to multiprocessing because PPE was not guaranteeing process termination.
    def timed_run(self, data, mrows=None, tgraph=None):

        n, d = data.shape
        timeout_seconds = 60 * 60  # 1 minute per variable
        queue = mp.Queue()
        p = mp.Process(target=_timed_run_wrapper, args=(queue,self, data, mrows, tgraph))
        p.start()

        try:
            status, payload = queue.get(timeout=timeout_seconds)
            p.join()
            if status == "ok":
                return payload
            else:
                raise RuntimeError(f"Exception in subprocess:\n{payload}")
        except Exception as e:
            # Timeout or other queue issues
            if p.is_alive():
                print("Method timeout... killing worker.")
                p.terminate()
                p.join()
                return None
        finally:
            # Ensure resources are cleaned
            p.close()
            queue.close()
            queue.join_thread()

        return None


    def ttimed_run(self,data,mrows=None,tgraph=None):
        n,d   = data.shape        
        #scale allowed computation time with dims
        #5 nodes: 5 minutes
        #10 nodes: 10 minutes
        #15 nodes: 15 minutes

        timeout_seconds = d * 60 #1 minute per variable


        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.run,data,mrows,tgraph)
            try:
                idata = future.result(timeout=timeout_seconds)

                return idata
            except concurrent.futures.TimeoutError:
                print("Method timeout...")
                future.cancel()  # attempt to cancel
                executor.shutdown(wait=False, cancel_futures=True)

                #executor.shutdown(cancel_futures=True)  # kills running processes

                return None

        return None

    def run(self,data,mrows=None,tgraph=None):
        raise NotImplementedError("Run must be implemented using a subclass")



#LOGIC Imports
from logic.logic import LOGIC
from logic.perfect_oracle import PerfectOracle
from logic.mdl_oracle import MDLOracle
from logic.globe_oracle import GLOBEOracle
from logic.poly_oracle import PolyOracle
from logic.mdl_score import MDLScore
from logic.poly_score import PolyScore
from logic.globe_score import GLOBEScore
from sklearn.metrics import f1_score, average_precision_score,accuracy_score

class LOGICX(BaseMethod):
    def __init__(self,oracle=True):
        self.oracle = oracle
        self.name_  = "LOGICG " if not oracle else "LOGICO "
        self.iscpdag = False

    def run(self,data,mrows=None,tgraph=None):
        miss_data,graph = None,None

        
        indep_oracle    = PerfectOracle(tgraph) if self.oracle is True else GLOBEOracle()
        if self.oracle: 
            print("I will use an oracle for the source selection phase")
        else:
            print("I will not use an oracle for first phase")
            
        order_score     = GLOBEScore(cache_result=False)
        alg             = LOGIC(orc=indep_oracle,sc=order_score)
        miss_data,graph = alg.fit_transform(data)

        return miss_data,graph,self.iscpdag
    
    def source_run(self,data,src,mrows=None,tgraph=None):
        miss_data,graph = None,None

        indep_oracle    = PerfectOracle(tgraph) if self.oracle is True else GLOBEOracle()
        order_score     = GLOBEScore(cache_result=False)
        alg             = LOGIC(orc=indep_oracle,sc=order_score)
        pred        = alg.cluster_transform(data)
        
        #import ipdb;ipdb.set_trace()
        dims     = data.shape[1]
        src_bin  = np.zeros(dims)
        pred_bin = np.zeros(dims)
        
        for i in range(len(src)):
            src_bin[src[i]]=1
            
        for i in range(len(pred)):
            pred_bin[pred[i]]=1
            
        auprc = np.round(average_precision_score(src_bin, pred_bin),2)
        f1 = np.round(f1_score(src_bin, pred_bin),2)
        acc = np.round(accuracy_score(src_bin, pred_bin),2)

        return src,pred,auprc,f1,acc
