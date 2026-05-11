import numpy as np

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