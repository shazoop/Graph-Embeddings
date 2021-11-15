import numpy as np
import torch
import random

def embed_to_device(nbd_embeddings, ent_code, rel_code, device):
    ent_embeddings = torch.from_numpy(np.stack(list(ent_code.values()),0))
    rel_embeddings = torch.from_numpy(np.stack(list(rel_code.values()),0))
    ent_embeddings = ent_embeddings.to(device)
    rel_embeddings = rel_embeddings.to(device)
    nbd_embeddings = torch.from_numpy(nbd_embeddings).to(device) 
    return nbd_embeddings, ent_embeddings, rel_embeddings

def tch_projmatrix(ent, ent_embeddings):
    '''
    Given the code for an entity h, will return a batch of projection matrices for each entity e.
    If h,e are vector embeddings, then will compute eh^T for each entity e. Returns the batch of these matrices.
    
    Input: ent (vector embedding), dictionary of entity vector embeddings
    
    Output: batch of projection matrices, one for each entity
    '''
    return torch.einsum('ni,j ->nij',ent_embeddings,ent)

def tch_score(edge,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor= False):
    '''
    Score the proposed relation (h,r,t) by using similarity matching. Assume tail is the variable.
    we just use connectivity here and ignore relations when computing similarity
    edges is a tuple (mid1, relation, mid2)
    '''
    #Get the ids/vector embeddings for head, tail, relation. Get nbd embedding for head
    head, relation ,tail = edge[0], edge[1], edge[2]
    head_ix, tail_ix, relation_ix = ent_id[head], ent_id[tail], rel_id[relation]
    head_embed, tail_embed, rel_embed = ent_embeddings[head_ix], ent_embeddings[tail_ix], rel_embeddings[relation_ix]
    
    if rel_tensor == True:
        head_nbd = nbd_embeddings[head_ix,:,:,0] #just use edge connectivity
        nbd_embed_norel = nbd_embeddings[:,:,:,0]
    else:
        head_nbd = nbd_embeddings[head_ix]
        nbd_embed_norel = nbd_embeddings
    
    #Generate the projection matrices for the given head
    proj = tch_projmatrix(head_embed, ent_embeddings)
    Frob_norm_sq = (torch.norm(head_nbd)**2).cpu().item()
    #Generate the 
    
    #Compute the graph homomorphism coeff
    x1 = torch.einsum('nij,jk -> nik',proj,head_nbd) #left multiply by projection matrix
    x2 = torch.einsum('ij,nkj -> nik',head_nbd,proj) #right multiply by transpose projection
    x = x1 + x2
    res = torch.bmm(x.permute(0,2,1),nbd_embed_norel)
    coeff = (1/(Frob_norm_sq + 1e-3))*torch.einsum('nii -> n',res) #number of matching edges
    
    #Get the top k
    (val,ix) = torch.topk(coeff,k)
    edge_coeff = torch.zeros(k-1)
    for i in range(1,k): #ignore top match, since that's likely to be nbd_embedding of head itself
        curr_ix = ix[i].cpu().item()
        edge_coeff[i-1] = torch.einsum('i,ij,j->',ent_embeddings[curr_ix],nbd_embeddings[curr_ix],tail_embed).cpu().item()
#         edge_coeff = 2*edge_coeff.clamp(min=0.,max=1.) - 1
    
    #Compute score by weighting each score (-1,1) by softmax of the similarities
    score = torch.dot(val[1:].float().cpu(),edge_coeff.cpu())

    
    return score, val, edge_coeff, Frob_norm_sq
    
def test(num_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, edge_set, rel_tensor= False):
    edges = list(edge_set['test.txt'])
    batch = random.sample(edges,num_batch)
    ttl_score = 0
    num_edges = 0
    
    for e in batch:
        score, _ , _, Frob_norm_sq = tch_score(e,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor)
        if Frob_norm_sq <= 1e-3:
            continue
        ttl_score = ttl_score + score.cpu().item()
        num_edges = num_edges + 1
    
    return ttl_score/num_edges
    