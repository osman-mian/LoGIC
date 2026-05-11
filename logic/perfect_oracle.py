import numpy as np
import networkx as nx
from logic.mdl_score import MDLScore

class PerfectOracle:
    
    def __init__(self,graph):
        self.graph = graph
        self.score         = MDLScore(cache_result=False)
        self.order = self.find_layer_order()
        self.dims  = graph.shape[1]
        self.indp  = graph * 0
        self.G = nx.DiGraph(self.graph)
        self.name_ = 'Perfect Oracle'

        
        for i in range(self.dims):
            for j in range(self.dims):
                self.indp[i,j]=-1

    def init_independence(self,dims):
        return
        self.indp = np.eye(dims)*0
        self.dims = dims
        for i in range(self.dims):
            for j in range(self.dims):
                self.indp[i,j]=-1            
                
    def is_independent(self,x,y,data):
        #print(self.indp[x,y])
        if self.indp[x,y] ==-1:
            self.indp[x,y] = nx.d_separated(self.G,{x},{y},{})
            self.indp[y,x] = self.indp[x,y]
        return self.indp[x,y]

        
    def find_layer_order(self):
        graph = self.graph
        queue = []
        visited = [False for i in range(graph.shape[1])]
        layer = 1
        sources = np.argwhere(np.sum(graph,axis=0)==0)

        #all sources go in by default as layer 1
        for src in sources:
            queue.append( (src.item(),layer) )

        #print(queue)
        while len(queue)!=0:
            node, layer = queue.pop(0)
            visited[node] = layer
            children = np.argwhere(graph[node,:]!=0)
            for c in children: queue.append((c.item(),layer+1))

        if False in visited: print("Error in traversal, atleast one node was left unvisited")
        
        return visited
    
'''----- Test stub ------    
def main():
    graph = np.eye(5) * 0
    
    a,b,c,d,e = 0,1,2,3,4
    header={0:'a',1:'b',2:'c',3:'d',4:'e'}
    
    graph[a,c]=1
    graph[b,c]=1
    graph[b,d]=1
    graph[d,e]=1
    
    orc = PerfectOracle(graph)
    #print(orc.order)
    
    for i in range(5):
        for j in range(i+1,5):
            print(header[i]," -> ",header[j],": ", orc.is_independent(i,j)," ... ",header[j]," -> ",header[i],": ",orc.is_independent(j,i))
    
main()
#'''