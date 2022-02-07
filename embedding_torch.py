import numpy as np
import torch
import random
import matplotlib.pyplot as plt

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
    
    #Compute the graph homomorphism coeff
    x1 = torch.einsum('nij,jk -> nik',proj,head_nbd) #left multiply by projection matrix
    x2 = torch.einsum('ij,nkj -> nik',head_nbd,proj) #right multiply by transpose projection
    x = x1 + x2
    res = torch.bmm(x.permute(0,2,1),nbd_embed_norel)
    coeff = (1/(Frob_norm_sq + 1e-3))*torch.einsum('nii -> n',res) #number of matching edges
    
    #Get the top k
    (val,ix) = torch.topk(coeff,k+1)
#     edge_coeff = torch.zeros(k-1)
#     for i in range(1,k): #ignore top match, since that's likely to be nbd_embedding of head itself
#         curr_ix = ix[i].cpu().item()
#         edge_coeff[i-1] = torch.einsum('i,ij,j->',ent_embeddings[curr_ix],nbd_embeddings[curr_ix],tail_embed).cpu().item()
#         edge_coeff = 2*edge_coeff.clamp(min=0.,max=1.) - 1
    edge_coeff = torch.zeros(k)
    for i in range(1,k+1): #ignore top match, since that's likely to be nbd_embedding of head itself
        curr_ix = ix[i].cpu().item()
        edge_coeff[i-1] = torch.einsum('i,ij,j->',ent_embeddings[curr_ix],nbd_embeddings[curr_ix],tail_embed).cpu().item()
    edge_coeff = 2*edge_coeff - 1
    
    #Compute score by weighting each score (-1,1) by softmax of the similarities
#     score = torch.dot(torch.nn.functional.softmax(val[1:], dim = 0).float().cpu(),edge_coeff.cpu())
    score = torch.dot(torch.nn.functional.softmax(val[1:], dim = 0).float().cpu(),edge_coeff.cpu())

    
    return score, val, edge_coeff, Frob_norm_sq
#     return score, val, ix, edge_coeff, Frob_norm_sq

# def tch_score_weight(edge,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, max_edges, rel_tensor= False):
#     '''
#     Score the proposed relation (h,r,t) by using similarity matching. Assume tail is the variable.
#     we just use connectivity here and ignore relations when computing similarity
#     edges is a tuple (mid1, relation, mid2)
#     '''
#     #Get the ids/vector embeddings for head, tail, relation. Get nbd embedding for head
#     head, relation ,tail = edge[0], edge[1], edge[2]
#     head_ix, tail_ix, relation_ix = ent_id[head], ent_id[tail], rel_id[relation]
#     head_embed, tail_embed, rel_embed = ent_embeddings[head_ix], ent_embeddings[tail_ix], rel_embeddings[relation_ix]
    
#     if rel_tensor == True:
#         head_nbd = nbd_embeddings[head_ix,:,:,0] #just use edge connectivity
#         nbd_embed_norel = nbd_embeddings[:,:,:,0]
#     else:
#         head_nbd = nbd_embeddings[head_ix]
#         nbd_embed_norel = nbd_embeddings
    
#     #Generate the projection matrices for the given head
#     proj = tch_projmatrix(head_embed, ent_embeddings)
#     Frob_norm_sq = (torch.norm(head_nbd)**2).cpu().item()
    
#     #Compute the graph homomorphism coeff
#     x1 = torch.einsum('nij,jk -> nik',proj,head_nbd) #left multiply by projection matrix
#     x2 = torch.einsum('ij,nkj -> nik',head_nbd,proj) #right multiply by transpose projection
#     x = x1 + x2
#     res = torch.bmm(x.permute(0,2,1),nbd_embed_norel)
#     coeff = (1/(Frob_norm_sq + 1e-3))*torch.einsum('nii -> n',res) #number of matching edges
    
#     #Get the top k
#     (val,ix) = torch.topk(coeff,k+1)
#     edge_coeff = torch.zeros(k)
#     for i in range(1,k+1): #ignore top match, since that's likely to be nbd_embedding of head itself
#         curr_ix = ix[i].cpu().item()
#         edge_coeff[i-1] = torch.einsum('i,ij,j->',ent_embeddings[curr_ix],nbd_embeddings[curr_ix],tail_embed).cpu().item()
#         edge_coeff = 2*edge_coeff - 1

    
#     #Compute score by weighting each score (-1,1) by softmax of the similarities
#     score = (Frob_norm_sq/max_edges) * torch.dot(torch.nn.functional.softmax(val[1:], dim = 0).float().cpu(),edge_coeff.cpu())

    
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

def test_weight(num_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, edge_set, max_edges, rel_tensor= False):
    edges = list(edge_set['test.txt'])
    batch = random.sample(edges,num_batch)
    ttl_score = 0
    num_edges = 0
    
    for e in batch:
        score, _ , _, Frob_norm_sq = tch_score_weight(e,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, max_edges, rel_tensor)
        if Frob_norm_sq <= 1e-3:
            continue
        ttl_score = ttl_score + score.cpu().item()
        num_edges = num_edges + 1
    
    return ttl_score/num_edges

def batch_plot(num_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, edge_set, norm_threshold = 0, rel_tensor= False):
    plot_array = np.zeros((num_batch,2))
    batch = random.sample(edge_set['test.txt'],num_batch)
    for (i,e) in enumerate(batch):
        score, _, _, norm = tch_score(e,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, 5, rel_tensor= False)
        plot_array[i,1] = score.numpy().item()
        plot_array[i,0] = norm
    plot_array = plot_array[plot_array[:,0] > norm_threshold]
    x, y = plot_array[:,0], plot_array[:,1]
    m,b = np.polyfit(x,y,1)
    plt.plot(x,y, '.')
    plt.plot(x,m*x + b)
    plt.show()
    
# def RecipRank(edge,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor= False):
#     head, relation ,tail = edge[0], edge[1], edge[2]
#     head_ix = ent_id[head]
    
#     if rel_tensor == True:
#         head_nbd = nbd_embeddings[head_ix,:,:,0] #just use edge connectivity
#     else:
#         head_nbd = nbd_embeddings[head_ix]
#     Frob_norm_sq = (torch.norm(head_nbd)**2).cpu().item()
    
#     score_vec = np.zeros(len(ent_id))
#     for ent in list(ent_id.keys()):
#         ent_ix = ent_id[ent]
#         new_edge = (ent, relation, tail)
#         score, _, _, _ = tch_score(new_edge,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor)
#         score_vec[ent_ix] = score
#     order = score_vec.argsort()
#     ranks = order.argsort()
#     return 1/ranks[head_ix], Frob_norm_sq

def batch_RR(edge,num_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor= False, reverse = False):
    head, relation ,tail = edge[0], edge[1], edge[2]
    head_ix = ent_id[head]
    if rel_tensor == True:
        head_nbd = nbd_embeddings[head_ix,:,:,0] #just use edge connectivity
    else:
        head_nbd = nbd_embeddings[head_ix]
    Frob_norm_sq = (torch.norm(head_nbd)**2).cpu().item()

    batch = random.sample(list(ent_id.keys()), num_batch-1)
    score_vec = np.zeros(num_batch)
    
    for (i,ent) in enumerate(batch):
        ent_ix = ent_id[ent]
        if reverse == True:  
            new_edge = (tail, relation, ent)
        else:
            new_edge = (ent, relation, tail)
        score, _, _, _ = tch_score(new_edge,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor)
        score_vec[i+1] = score
    score, _, _, _ = tch_score(edge,ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor)
    score_vec[0] = score
    score_vec = -1*score_vec
    order = score_vec.argsort()
    ranks = order.argsort()
    ranks  = ranks + 1
    return 1/(ranks[0]), Frob_norm_sq

def batch_MRR(num_batch, num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, edge_set, rel_tensor= False, reverse = False):
    batch = random.sample(edge_set['test.txt'],num_batch)
    ttl_RR = 0
    
    for e in batch:
        RR, _  = batch_RR(e,num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor, reverse)
        ttl_RR = ttl_RR + RR
    
    return ttl_RR/num_batch

def batch_MRR_thresh(num_batch, num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, edge_set, threshold = 0., rel_tensor= False, reverse = False):
    batch = random.sample(edge_set['test.txt'],num_batch)
    ttl_RR = 0
    N = 0
    for e in batch:
        head, rel, tail = e[0], e[1], e[2]
        ent_ix = ent_id[head]
        ent_nbd = nbd_embeddings[ent_ix]
        norm = (torch.norm(ent_nbd)**2).cpu().item()
        if norm > threshold:
            RR, _ = batch_RR(e,num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor, reverse)
            ttl_RR = ttl_RR + RR
            N += 1
    
    return ttl_RR/N

def RR_plot(num_batch, num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, edge_set, norm_threshold = 0, rel_tensor= False, reverse = False):
    plot_array = np.zeros((num_batch,2))
    batch = random.sample(edge_set['test.txt'],num_batch)
    
    for (i,e) in enumerate(batch):
        RR, norm  = batch_RR(e,num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor, reverse)
        plot_array[i,1] = RR
        plot_array[i,0] = norm
    plot_array = plot_array[plot_array[:,0] > norm_threshold]
    x, y = plot_array[:,0], plot_array[:,1]
    m,b = np.polyfit(x,y,1)
    plt.plot(x,y, '.')
    plt.plot(x,m*x + b)
    plt.show()


def batch_hit(num_batch, num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, edge_set, K, rel_tensor= False, reverse = False):
    batch = random.sample(edge_set['test.txt'],num_batch)
    ttl_hit = 0
    
    for e in batch:
        RR, _  = batch_RR(e,num_mini_batch, ent_embeddings, rel_embeddings, nbd_embeddings, ent_id, rel_id, k, rel_tensor, reverse)
        if RR >= 1/K:
            ttl_hit += 1
    
    return ttl_hit/num_batch
