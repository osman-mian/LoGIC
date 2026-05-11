import random
import numpy as np


def chain(nodes=5):
    n_id = [i for i in range(nodes)]
    #random.shuffle(n_id)
    graph = np.eye(nodes)*0
    
    for i in range(1,nodes):  #iterating over indexes, not actual nodes
        pa = n_id[i-1]
        ch = n_id[i]
        graph[pa,ch]=1
        
        
    return graph

def full(nodes=5):
    n_id = [i for i in range(11)]
    graph = np.eye(nodes)*0
    
    A,B,C,D,E,F,G,H,I,J,K = 0,1,2,3,4,5,6,7,8,9,10
    edges=[(A,C),(A,D),(B,E),(B,K),(C,F),(C,G),(D,H),(E,H),(E,I),(F,G),(H,J),(I,K)]
    
    for edge in edges:
        pa = edge[0]
        ch = edge[1]
        graph[pa,ch]=1
        
    return graph

def river(nodes=12):
    #print("river")
    branch_size = nodes//4
    nodes = nodes
    
    n_id = [i for i in range(nodes)]
    #random.shuffle(n_id)
    graph = np.eye(nodes)*0
    
    #two branches merging into a a third branch via collider.
    
    
    #first chain
    #print("1st...")
    start1 =    0
    end1   =    start1 + branch_size
    for i in range(start1+1,end1):
        pa = n_id[i-1]
        ch = n_id[i]
        graph[pa,ch]=1
        #print(pa,"->",ch)

    #print("2nd...")   
    #second chain
    start2 =    branch_size * 1
    end2   =    start2 + branch_size
    for i in range(start2+1,end2):
        pa = n_id[i-1]
        ch = n_id[i]
        graph[pa,ch]=1
        #print(pa,"->",ch)
    
    #print("3rd...")
    #third chain
    start3 =    branch_size * 2
    end3   =    start3 + branch_size
    for i in range(start3+1,end3):
        pa = n_id[i-1]
        ch = n_id[i]
        graph[pa,ch]=1


    start4 =    branch_size * 3
    end4   =    nodes
    for i in range(start4+1,end4):
        pa = n_id[i-1]
        ch = n_id[i]
        graph[pa,ch]=1
        
        
    #print("Merge...")
    #make a collider with first two chains merging into start of the third
    pa1 = n_id[end1-1]
    pa2 = n_id[end2-1]
    pa3 = n_id[end3-1]
    ch  = n_id[start4]
    #print(pa1,"->",ch)
    #print(pa2,"->",ch)
    graph[pa1,ch]=1
    graph[pa2,ch]=1
    graph[pa3,ch]=1
    #print(graph)
    return graph
    
    
    
    
        
    

    

def tree(nodes=10):
    n_id = [i for i in range(nodes)]
    #random.shuffle(n_id)
    graph = np.eye(nodes)*0
    
    
    num_branches = np.random.randint(2,4) #anywhere between 2,3 branches
    
    #choose a root
    root = n_id.pop(0)
    
    #tree is just multiple chains, each starting with the same root
    branches = np.array_split(np.array(n_id,dtype=int),num_branches)
    
    #iterate over each branch
    for i in range(num_branches):
        temp_branch = branches[i].tolist()  #pick a branch
        pa = root                           #root is always the first parent
        for tnode in temp_branch:            #for each element of the branch, iterating over nodes, not over indexes
            ch = tnode                      #treat it as a child of the parent (from previous iteration, 0th iteration parent is root)
            graph[pa,ch]=1                  #set edge to true
            pa = tnode                       #this child is the parent for next node in branch
            
    return graph

def collider(nodes=5):
    n_id = [i for i in range(nodes)]
    #random.shuffle(n_id)
    graph = np.eye(nodes)*0
    
    for i in range(len(n_id)-1):
        graph[i,len(n_id)-1]=1
    
    return graph

def diamond(nodes):
    n_id = [i for i in range(5)]
    #random.shuffle(n_id)
    graph = np.eye(len(n_id))*0

    # First node (0) points to two nodes
    # then those two nodes (1,2) both go into a fourth node (3)
    # the fourth node has a single descendent (4)
    
    graph[n_id[0],n_id[1]]=1
    graph[n_id[0],n_id[2]]=1
    graph[n_id[1],n_id[3]]=1
    graph[n_id[2],n_id[3]]=1
    graph[n_id[3],n_id[4]]=1
    
    return graph

'''------code for sanity testing --------
def print_g(g):
    dims = g.shape[1]
    for i in range(dims):
        for j in range(dims):
            if g[i,j]==1: print(i,"->",j)
    print("-------")
    
    
def main():
    print("Chain...")
    for i in range(3,8):
        print_g(chain(i))
    print("======")

    print("Tree...")
    for i in range(3,8):
        print_g(tree(i))
    print("======")

    print("Diamond...")
    for i in range(3,8):
        print_g(diamond(i))
    print("======")

    print("Collider...")
    for i in range(3,8):
        print_g(collider(i))
    print("======")

main()
#'''