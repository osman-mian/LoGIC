from functools import total_ordering
import numpy as np

@total_ordering
class Node:
    
    def __init__(self,id,dims):
        self.id=id
        self.comps = np.zeros(dims)
        
    def __lt__(self, other):
        source = self.id
        target = other.id
        
        # we check if gain(self -> target)  >? gain(target->self) 
        #true means self node is in an upper layer than target, upper layer is indexed by a smaller number, hence self should be considered less than target
        #false means self node is in a lower layer than target, lower layer is indexed by a larger number,  hence self should be considered more than target
        #delta = self.comps[target] > other.comps[source] # higher gains should get lower index, therefore we compare using gt sign inside a lt function
        
        delta = np.sum(self.comps) > np.sum(other.comps) #for this check we are simply seeing how many times was this node ranked better. Should be higher for "sources"
        
        return delta
    
    
    def __eq__(self, other):
        source = self.id
        target = other.id
        #delta  = self.comps[target] == other.comps[source]  
        delta = np.sum(self.comps) ==np.sum(other.comps)
        return delta

'''
from random import shuffle
def main():
    dims = 5
    node_array=[]
    for i in range(dims):
        node_array.append(Node(i,dims))
        
    #graph === C->B, B->A, C->D, D->E
    A,B,C,D,E = 0,1,2,3,4
    
    node_array[C].comps[B] = 40  #C->B
    node_array[B].comps[C] = -40  
    
    node_array[C].comps[A] = 30  #C->A (ancestral)
    node_array[A].comps[C] = 10  
    
    node_array[B].comps[A] = 50  #B->A
    node_array[A].comps[B] = 40  
    
    node_array[C].comps[D] = 20  #C->D
    node_array[D].comps[C] = 10  
    
    node_array[C].comps[E] = 15  #C->E (ancestral)
    node_array[E].comps[C] = 5    
    
    node_array[D].comps[E] = 30  #D->E
    node_array[E].comps[D] = 20  
    
    node_array[D].comps[A] = 5  #D->A (backdoor)
    node_array[A].comps[D] = 1  
    
    node_array[B].comps[E] = 10  #B->E (backdoor)
    node_array[E].comps[B] = 4  
    
    node_array[E].comps[A] =  5  #E->A (same depth)
    node_array[A].comps[E] = 6  
    
    node_array[D].comps[B] = 8  #D->B (same depth)
    node_array[B].comps[D] = 11  
    
    for i in range(10):
        orig = [n.id for n in node_array]
        shuffle(orig)
        node_array.sort()
        mod  = [n.id for n in node_array]
    
        print(orig)
        print(mod)
        print("--------")
    
main()
#'''