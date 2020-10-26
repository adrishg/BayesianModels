import numpy as np
import networkx as nx
from itertools import product 

########################
##### Tools Section ####
def print_readable_adj(H):
    """This function print the adjaceny matriz substituing the 1s for the variable letter. """
    #todense: Return a dense matrix representation of this matrix.
    A = nx.adjacency_matrix(H).todense()
    for n in H.nodes:
        print(H.nodes[n]['name'],[ str(i) if i == 0 else H.nodes[c]['name'] for c,i in enumerate(np.array(A[n])[0])])  

def gen_table_shape(n,H):
    """This function sets the shape of the probability table for the "n" varible up.
       n is the node number. 
       It returns a numpy array of zeros of dimensions (x,y1,..,yk)
       where x is the number of states of variable n.
       k is the number of parents that variable n has.
       yk is the number of states that the parent k has.
       the parents must be ordered according to their node number (see "store_partents" function). 
       """
    states_parents = [ len(H.nodes[i]['states']) for i in H.nodes[n]['parents'] ]
    states_current = [len(H.nodes[n]['states'])]
    tab_shape = tuple(states_current+states_parents)
    return np.zeros(tab_shape)

def store_parents(H):
    """This function adds the information of the parents of a given node.
       This are stored in ascending order according to their node number."""

    for n in H.nodes:
        H.nodes[n]['parents'] = sorted(H.predecessors(n))
    return H    

def distribution(v,H):
    """This function computes the Probability value given a state vector."""
    term1 = H.nodes[3]['pbt'][v[3],v[0],v[1]]  # P(T|R,S)
    term2 = H.nodes[2]['pbt'][v[2],v[0]]       # P(J|R)
    term3 = H.nodes[1]['pbt'][v[1],0]          # P(S)
    term4 = H.nodes[0]['pbt'][v[0],0]          # P(R)
    return term1*term2*term3*term4   

def gen_state_space(H):
    """This function generates the list of state vectors"""
    return list(product(*[ range(np.shape(H.nodes[n]['pbt'])[0]) for n in H.nodes]))

def eval_dist(v,H):
    """ This function takes a vector state given H and computes its prob."""
    val = 0
    for n in H.nodes:
        parent_list = H.nodes[n]['parents']
        if len(parent_list):
           tar = H.nodes[n]['pbt'][v[n]]
           for p in parent_list:
               tar = tar[v[p]]
        else:
           tar = H.nodes[n]['pbt'][v[n],0]
        with np.errstate(divide='ignore'):
           val += -np.log(tar)
    return np.exp(-val)   

def compute_cond_prob(sttspc,H,infer,given):
    """This function computes de conditiona probability over the entire distribution.
       sttspc space is state_space, 
       H is the graph , 
       infer is a list of tuples (Var,VarState) for which the probibility are being computed.
       given is a list of tuples (Var,VarState) for the evidence variables.
           P(infer_variables|evidence_variables)"""
    deno = 0
    nume = 0
    for stt in sttspc:
        if all(stt[i[0]] is i[1] for i in given):
            deno += eval_dist(stt,H)
            if all(stt[i[0]] is i[1] for i in infer):
               nume += eval_dist(stt,H)
    return nume/deno

##### End of tools section ####
###############################
