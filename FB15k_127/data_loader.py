import numpy as np
import os
import math

def get_sets(path, edge_filter = False, max_edges = None):
    """
    Parses the raw FB text files and returns the set of all entities(their machine ids) and relations
    
    !!!The entities are sorted before assigning them to an index
    
    Input: path object
    
    Output: dictionaries of unique integer index to each entity and relation, set of edges for each split
    """
    entities, relations = set(), set()
    edge_set = {}
    for split in ["train.txt", "valid.txt", "test.txt"]:
        with open(os.path.join(path, split), "r") as lines:
            edges = set()
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
                edges.add((lhs,rel,rhs))
        edge_set[split] = edges
    
    ent_id = {k:i for (i,k) in enumerate(sorted(entities))}
    rel_id = {k:i for (i,k) in enumerate(sorted(relations))}
    
    if edge_filter is True:
        assert max_edges != None, 'need to specify maximum number of edges when filtering'
        connect_count = {k:0 for k in list(ent_id.keys())}
        with open(os.path.join(path, 'train.txt'), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                connect_count[lhs] = connect_count[lhs] + 1
                connect_count[rhs] = connect_count[rhs] + 1

        keep_ent = [k for k,v in connect_count.items() if v <= max_edges]
        ent_id = {k:i for i,k in enumerate(keep_ent)}
        edge_set = {}
        for split in ["train.txt", "valid.txt", "test.txt"]:
            with open(os.path.join(path, split), "r") as lines:
                edges = set()
                for line in lines:
                    lhs, rel, rhs = line.strip().split("\t")
                    if (lhs in keep_ent) and (rhs in keep_ent):
                        edges.add((lhs,rel,rhs))
            edge_set[split] = edges

               
    return ent_id, rel_id, edge_set

def get_codebook(path, D_ent, D_rel, edge_filter = False, max_edges = None):
    """
    Generates a codebook for the set of entities and relations by randomly sampling from the unit hyperspheres of dimension
    D_ent and D_rel respectively. 
    
    input: path object, entity embedding dimension, relation embedding dimension
    
    output: dictionary of embeddings for entity and relation set (machine ids are keys for entity), plus output from get_id
    """
    ent_id, rel_id, edge_set = get_sets(path, edge_filter, max_edges)
    
    def normal_vec(dim):
        vec = np.random.multivariate_normal(np.zeros(dim),np.eye(dim))
        return vec/np.linalg.norm(vec)
    
    def nv_append1(dim):
        vec = np.random.multivariate_normal(np.zeros(dim),np.eye(dim))
        return np.append(1.,vec/np.linalg.norm(vec))

    
    ent_code = {k:normal_vec(D_ent) for k in list(ent_id.keys())}
    rel_code = {k:nv_append1(D_rel) for k in list(rel_id.keys())}
    
    return ent_id, rel_id, ent_code, rel_code, edge_set

def get_embeddings(split, ent_id, rel_id, ent_code, rel_code, edge_set, rel_tensor = False):
    """
    For each entity, generate the embedding of its neighborhood subgraph (all edges involving the entity).
    
    input: path object, data split, entity embedding dim, relation embedding dim
    """
    N_ent = len(ent_id)
    D_ent = (list(ent_code.values())[0]).shape[0]
    D_rel = (list(rel_code.values())[0]).shape[0]
    
    if rel_tensor == True:
        embeddings = np.zeros((N_ent,D_ent,D_ent,D_rel+1))
    else:
        embeddings = np.zeros((N_ent,D_ent,D_ent))
    
    for e in list(edge_set[split]):
        head, rel, tail = e[0], e[1], e[2]
        head_ebd, rel_ebd, tail_ebd = ent_code[head], rel_code[rel], ent_code[tail]
        if rel_tensor == True:
            tensor = np.einsum('i,j,k -> ijk',head_ebd,tail_ebd,rel_ebd)
        else:
            tensor = np.einsum('i,j -> ij', head_ebd,tail_ebd)
        embeddings[ent_id[head]] += tensor
        embeddings[ent_id[tail]] += tensor
                
    return embeddings

def easy_embeddings(path, split, D_ent, D_rel, rel_tensor = False, edge_filter = False, max_edges = None):
    """
    For each entity, generate the embedding of its neighborhood subgraph (all edges involving the entity).
    
    input: path object, data split, entity embedding dim, relation embedding dim
    """
    ent_id, rel_id, ent_code, rel_code, edge_set = get_codebook(path, D_ent, D_rel, edge_filter, max_edges)
    
    embeddings =   get_embeddings(split, ent_id, rel_id, ent_code, rel_code, edge_set, rel_tensor)
    
    return embeddings, ent_id, rel_id, ent_code, rel_code, edge_set

def avg_ePn(edge_set):
    '''
    Estimates average edges per node by just dividing number of edges by number of nodes
    Input is the 'edge_set' output from any of the "get_..." functions
    '''
    edge_total = 0
    for key in list(edge_set.keys()):
        edge_total = edge_total + len(edge_set[key])
    return(math.ceil(edge_total/len(ent_id)))