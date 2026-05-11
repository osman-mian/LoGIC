import pandas as pd
import pickle 
import pandasql as ps

class Experiment:
    
    def __init__(self,headers,fname,preload=False):
        self.exp_list = pd.DataFrame(columns=headers)
        self.fname    = fname
        if preload:
            self.load_state(fname)
            print("Loaded: ", self.exp_list.shape)
        
    def add(self, exp):
        self.exp_list.loc[-1]  = exp  # adding a row
        self.exp_list.index    = self.exp_list.index + 1  # shifting index
        self.exp_list          = self.exp_list.sort_index()  # sorting by index
        
    def save_state(self):
        #print(self.exp_list)
        with open(self.fname, 'wb') as handle:
            pickle.dump(self.exp_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 
    
    def load_state(self,fname):
        self.fname = fname
        with open(fname, 'rb') as handle:
            self.exp_list = pickle.load(handle)

    def query(self,query):
        results = self.exp_list
        res = ps.sqldf(query, locals())
        
        return res
        
        