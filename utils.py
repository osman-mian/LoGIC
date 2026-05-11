import numpy as np


def load_data(fname,delim=',',mv='?'):
    try:
        dx = np.genfromtxt(fname,delimiter=delim,missing_values=mv)
        return dx
    except FileNotFoundError as e:
        print("Error: ",e)
        return None
    
def load_graph(fname):
    return load_data(fname)

def clean(data,lim=3):
    mu_ = np.mean(data,axis=0);
    sd_ = np.std(data,axis=0);

    upper_limit= mu_ + lim*sd_;
    lower_limit= mu_ - lim*sd_;

    z1 = data[:,]<= upper_limit 
    z2 = data[:,]>=lower_limit;

    k1 =[];
    for r in range(len(z1)):
        k1.append( z1[r].all() and z2[r].all());

    return data[np.where(k1)];

def standardize(X,cleanup=True):
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    if not cleanup: return X_std
    return clean(X_std)

def binary_sampler(p: float, rows: int, cols: int) -> np.ndarray:
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

def rmse_loss(ori_data: np.ndarray, imputed_data: np.ndarray, data_m: np.ndarray,subMask=False) -> np.ndarray:
    rows         = ori_data.shape[0]
    imputed_data = imputed_data[0:rows,:] #remove last row because it was dummy added to make MIRACLE work
    data_m       = data_m[0:rows,:]       #same for mask
    
    filters   = np.isnan(imputed_data)
    corrector = np.sum(filters)
    
    nMask = np.copy(data_m)
    if subMask:
        nMask[filters] = 1
    
    #if the algorithm said "i dont know" for an imputation, assign it the value of true imputation and correct for rmse in denom
    imputed_data[filters] = ori_data[filters]
    
    numerator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)-corrector
    ir = np.round(denominator / (0.001+np.sum(1 - data_m)) ,2)
    #print("Correction factor: ",corrector)
    if denominator <=0: return -1,0
    rms = np.sqrt(numerator / float(denominator))
    

    if subMask:
        return np.round(rms,3),np.round(ir,3),nMask
    
    return np.round(rms,3),np.round(ir,3)
    


from cdt.metrics import SID, SID_CPDAG
from sep_calc import calc_sep_dist_dag,calc_sep_dist_cpdag
def calc_stats(pred,graph,cpdag=True):
    pred  = np.copy(pred)
    graph = np.copy(graph)
    dims  = pred.shape[0]

    if cpdag:
        paid,aaid   = calc_sep_dist_cpdag(pred,graph)
        sid_l,sid_h = SID_CPDAG(graph,pred)
        sid         = np.floor(0.5*(sid_l + sid_h).item())
        pred        = np.array(np.array(pred+pred.T,dtype=bool),dtype=int)
        graph       = np.array(np.array(graph+graph.T,dtype=bool),dtype=int)
        shd         = np.sum(np.logical_xor(pred,graph)) * 0.5 
        tp          = np.sum(pred*graph) * 0.5
        fp          = np.sum (pred * (1-graph)) * 0.5
        fn          = np.sum( (1-pred) * graph) * 0.5
    else:
        paid,aaid   = calc_sep_dist_dag(pred,graph)
        sid         = SID(graph,pred)
        shd         = np.sum(np.logical_xor(pred,graph))
        tp          = np.sum(pred*graph)
        fp          = np.sum (pred * (1-graph))
        fn          = np.sum( (1-pred) * graph)

    
    return shd,sid,paid,tp,fp,fn

def calc_ud_stats(pred,graph,cpdag=True):
    shd,sid,paid,tp,fp,fn = calc_stats(pred,graph,cpdag)
    return shd, sid,paid, tp, fp, fn


import matplotlib.pyplot as plt
def save_fig(x,y,fname):
    plt.figure()  # Start a new figure
    plt.scatter(x, y)
    plt.grid(True)
    plt.tight_layout()
    plt.title(fname)
    plt.savefig(fname)
    plt.close() 

    
def save_fig_tup(tup,fname):
    plt.figure()  # Start a new figure
    
    for t in tup:
        x,y,z = t[0],t[1],t[2]
        plt.scatter(x, y, c=z)
        plt.grid(True)
        plt.tight_layout()

    plt.title(fname)
    plt.savefig(fname)
    plt.close() 

    
    
def gaussian_cause(points):
    """Init a root cause with a Gaussian."""
    return np.random.randn(points, 1)[:, 0]


def test_graph(nodes=10,dgen=True,idx=0):
    
    if dgen:
        mech = 'sigmoid_add'
        generator = AcyclicGraphGenerator(mech, npoints=2000,nodes=nodes,parents_max=3,initial_variable_generator=gaussian_cause,noise_coeff=0.2)
        dt, G = generator.generate()
        graph = nx.to_numpy_array(G)
        data  = dt.to_numpy().reshape((-1,nodes))
        dims  = graph.shape[1]
        fname = './viz/sampled_'+mech+'/'+str(idx)+'/'

        data = standardize(data)
        if not os.path.exists(fname): os.makedirs(fname) 
        
    else:
        fname,gtname    = "./data/data"+str(idx)+".txt" , "./data/data"+str(idx)+"_truth.txt"
        data            = standardize(np.round(load_data(fname),3))
        graph           = load_graph(gtname)
        #print(fname)
        

    return fname,data,graph