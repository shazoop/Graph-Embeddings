# import torch
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math
from itertools import product

def vertex_code(dim, num):
    codebook = np.zeros((num,dim))
    for i in range(num):
        code = np.random.multivariate_normal(np.zeros(dim),np.eye(dim))
        code = code/np.linalg.norm(code)
        codebook[i] = code
    return(codebook)

def vertex_code_R(dim, num):
    codebook = np.zeros((num,dim))
    for i in range(num):
        code = 2*np.random.binomial(1,.5,dim)-1
        codebook[i] = code
    return(codebook)

def generate_edge(special_code):
    edge = np.einsum('i,j -> ij', special_code[0], special_code[1])
    return(edge)

def generate_edge_R(special_code):
    edge = special_code[0]*special_code[1]
    return(edge)

def generate_graph(codebook, num_edges):
    n,d = codebook.shape[0], codebook.shape[1]
    graph = 0
    for i in range(num_edges):
        d_ix, c_ix = rand.randint(0,n-1), rand.randint(0,n-1)
        dom, cod = codebook[d_ix], codebook[c_ix]
        graph  = graph + np.einsum('i,j -> ij', dom, cod)
    return(graph)

def generate_graph_R(codebook, num_edges):
    n,d = codebook.shape[0], codebook.shape[1]
    graph = 0
    for i in range(num_edges):
        d_ix, c_ix = rand.randint(0,n-1), rand.randint(0,n-1)
        dom, cod = codebook[d_ix], codebook[c_ix]
        graph  = graph + (dom*cod)
    return(graph)

def correct_edgeQ(codebook, special_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph(codebook,num_edges) + generate_edge(special_code)
        edgeQ = np.einsum('i,ij,j ->',special_code[0],curr_graph, special_code[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def correct_edgeQ_R(codebook, special_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph_R(codebook,num_edges) + generate_edge_R(special_code)
        edgeQ = np.sum(curr_graph*special_code[0]*special_code[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_edgeQ(codebook, special_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph(codebook,num_edges)
        edgeQ = np.einsum('i,ij,j ->',special_code[0],curr_graph, special_code[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_edgeQ_R(codebook, special_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph_R(codebook,num_edges)
        edgeQ = np.sum(curr_graph*special_code[0]*special_code[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def edge_composition(graph1,graph2):
    composed_graph = np.einsum('ij,jk -> ik',graph1,graph2)
    return(composed_graph)

def correct_compQ(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph(codebook,num_edges) + generate_edge(edge_code[:2]) + generate_edge(edge_code[1:3])
        curr_graph = edge_composition(curr_graph,curr_graph)
        edgeQ = np.einsum('i,ij,j ->',edge_code[0],curr_graph, edge_code[2])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_compQ(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph(codebook,num_edges)
        curr_graph = edge_composition(curr_graph,curr_graph)
        edgeQ = np.einsum('i,ij,j ->',edge_code[0],curr_graph, edge_code[2])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def edge_composition_R(graph1,graph2):
    composed_graph = graph1*graph2
    return(composed_graph)

def correct_compQ_R(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph_R(codebook,num_edges) + generate_edge_R(edge_code[:2]) + generate_edge_R(edge_code[1:3])
        curr_graph = edge_composition_R(curr_graph,curr_graph)
        edgeQ = np.sum(edge_code[0]*curr_graph*edge_code[2])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_compQ_R(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph = generate_graph_R(codebook,num_edges)
        curr_graph = edge_composition_R(curr_graph,curr_graph)
        edgeQ = np.sum(edge_code[0]*curr_graph*edge_code[2])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def testing_code(vertex_dim, vertex_num, book_size, num_edge, num_trials):
    special_code = vertex_code(vertex_dim,2)
    edge_code = vertex_code(vertex_dim,3)
    cor_edgeQ = []
    incor_edgeQ = []
    cor_compQ = []
    incor_compQ = []
    for size in book_size:
        codebook = vertex_code(vertex_dim,size)
        cor_edgeQ.append(correct_edgeQ(codebook, special_code, num_edge, num_trials))
        incor_edgeQ.append(incorrect_edgeQ(codebook, special_code, num_edge, num_trials))
        cor_compQ.append(correct_compQ(codebook, edge_code, num_edge, num_trials))
        incor_compQ.append(incorrect_compQ(codebook, edge_code, num_edge, num_trials))
    return cor_edgeQ, incor_edgeQ, cor_compQ, incor_compQ

def testing(vertex_dim, vertex_num, edge_list, num_trials):
    codebook = vertex_code(vertex_dim,vertex_num)
    special_code = vertex_code(vertex_dim,2)
    edge_code = vertex_code(vertex_dim,3)
    cor_edgeQ = []
    incor_edgeQ = []
    cor_compQ = []
    incor_compQ = []
    for num_edge in edge_list:
        cor_edgeQ.append(correct_edgeQ(codebook, special_code, num_edge, num_trials))
        incor_edgeQ.append(incorrect_edgeQ(codebook, special_code, num_edge, num_trials))
        cor_compQ.append(correct_compQ(codebook, edge_code, num_edge, num_trials))
        incor_compQ.append(incorrect_compQ(codebook, edge_code, num_edge, num_trials))
    return cor_edgeQ, incor_edgeQ, cor_compQ, incor_compQ

def testing_R(vertex_dim, vertex_num, edge_list, num_trials):
    codebook = vertex_code_R(vertex_dim,vertex_num)
    special_code = vertex_code_R(vertex_dim,2)
    edge_code = vertex_code_R(vertex_dim,3)
    cor_edgeQ = []
    incor_edgeQ = []
    cor_compQ = []
    incor_compQ = []
    for num_edge in edge_list:
        cor_edgeQ.append(correct_edgeQ_R(codebook, special_code, num_edge, num_trials))
        incor_edgeQ.append(incorrect_edgeQ_R(codebook, special_code, num_edge, num_trials))
        cor_compQ.append(correct_compQ_R(codebook, edge_code, num_edge, num_trials))
        incor_compQ.append(incorrect_compQ_R(codebook, edge_code, num_edge, num_trials))
    return cor_edgeQ, incor_edgeQ, cor_compQ, incor_compQ