import numpy as np
from logic.mdl_score import MDLScore

from causallearn.utils.cit import CIT

class MDLOracle:
    
    def __init__(self,alpha=0.01):
        self.score         = MDLScore(cache_result=False)
        self.sig_threshold = -np.log2(alpha)
        self.dims          = None
        self.indp          = None
        self.name_         = 'MDL     Oracle'
        #print(self.name_)
                
    def init_independence(self,dims):
        self.indp = np.eye(dims)*0
        self.dims = dims
        for i in range(self.dims):
            for j in range(self.dims):
                self.indp[i,j]=-1
        
    def is_independent(self,x,y,data):
        if self.indp[x,y] == -1:
            
            chunk = data[:,[x,y]]
            complete_chunk = chunk[~np.isnan(chunk).any(axis=1)]
            
            #kci_obj = CIT(complete_chunk, "kci",kernelZ='Polynomial', degree=5) # construct a CIT instance with data and method name
            #pValue = kci_obj(0, 1,None)
            #print(x,"/",y,": ",np.abs(np.max([delta_xy,delta_yx])))
            #indep = pValue < 0.05

            #in completed chunk, x will be 0th index and y will always be 1st index
            delta_xy = self.score.compute_gain(complete_chunk,1,[],0) #compute(data, i: int, PAi: List[int])
            delta_yx = self.score.compute_gain(complete_chunk,0,[],1)

            #to check if difference in compression is significant, if not we could declare that these variables don't have an edge between them
            indep = np.abs(np.max([delta_xy,delta_yx])) < self.sig_threshold
            #print(x,"/",y,": ",np.abs(np.max([delta_xy,delta_yx])))
            
            self.indp[x,y] = indep
            self.indp[y,x] = indep

        return self.indp[x,y]
        
        
        

    
    