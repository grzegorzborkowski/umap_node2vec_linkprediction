import networkx as nx
import scipy.sparse as sp
import numpy as np
from LP_arguments import LP_arguments
import argparse
from link_prediction_helpers import *
from collections import namedtuple
from models_factory import ModelFactory

MethodResult = namedtuple('MethodResult', ['methodName', 'testROC', 'testPC'])

def calculate(min_degree, file_path="graph.graph"):
    graph = nx.read_edgelist(file_path, delimiter=" ")
    nodes = [node for node, degree in graph.degree().items() if degree >= min_degree]
    graph = graph.subgraph(nodes)
    connected_components = nx.connected_components(graph)
    largest_cc_nodes = max(connected_components, key=len)
    graph = graph.subgraph(largest_cc_nodes)

    adj_sparse = nx.to_scipy_sparse_matrix(graph)
    adj = nx.to_numpy_matrix(graph)
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.1)

    g_train = nx.from_scipy_sparse_matrix(adj_train) # new graph object with only non-hidden edges
    aa_matrix = np.zeros(adj.shape)
    for u, v, p in nx.adamic_adar_index(g_train): # (u, v) = node indices, p = Adamic-Adar index
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p # make sure it's symmetric
    
    # Normalize array
    aa_matrix = aa_matrix / aa_matrix.max()
    aa_roc, aa_ap = get_roc_score(adj_sparse, test_edges, test_edges_false, aa_matrix)

    jc_matrix = np.zeros(adj.shape)
    for u, v, p in nx.jaccard_coefficient(g_train): # (u, v) = node indices, p = Jaccard coefficient
        jc_matrix[u][v] = p
        jc_matrix[v][u] = p # make sure it's symmetric
    
    jc_matrix = jc_matrix / jc_matrix.max()
    
    # Calculate ROC AUC and Average Precision
    jc_roc, jc_ap = get_roc_score(adj_sparse, test_edges, test_edges_false, jc_matrix)

    pa_matrix = np.zeros(adj.shape)
    for u, v, p in nx.preferential_attachment(g_train): # (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p # make sure it's symmetric
    
    # Normalize array
    pa_matrix = pa_matrix / pa_matrix.max()

    # Calculate ROC AUC and Average Precision
    pa_roc, pa_ap = get_roc_score(adj_sparse, test_edges, test_edges_false, pa_matrix)

    model_factory = ModelFactory(g_train)
    model = model_factory.get_model("node2vec")
    
    #TODO: refactor these three calls. Make a function out of it
    # Store embeddings mapping
    emb_mappings = model.wv
    emb_list = []
    for node_index in range(0, adj_sparse.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)
    
    umap_obj = model_factory.get_model("UMAP")
    emb_mappings_umap = umap_obj.fit_transform(emb_matrix)

    emb_list_umap = []
    for node_index in range(0, adj_sparse.shape[0]):
        node_emb = emb_mappings_umap[node_index]
        emb_list_umap.append(node_emb)
    emb_matrix_umap = np.vstack(emb_list_umap)

    pca_obj = model_factory.get_model("PCA")
    emb_mappings_pca = pca_obj.fit_transform(emb_matrix)

    emb_list_pca = []
    for node_index in range(0, adj_sparse.shape[0]):
        node_emb = emb_mappings_pca[node_index]
        emb_list_pca.append(node_emb)
    emb_matrix_pca = np.vstack(emb_list_pca)

    lp_arg = LP_arguments(emb_mappings=emb_mappings, adj_sparse = adj_sparse, train_edges = train_edges, \
     train_edges_false = train_edges_false, val_edges = val_edges, val_edges_false = val_edges_false, \
     test_edges = test_edges, test_edges_false = test_edges_false, matrix=emb_matrix)
    
    lp_arg_umap = LP_arguments(emb_mappings=emb_mappings_umap, adj_sparse=adj_sparse, train_edges = train_edges, \
     train_edges_false = train_edges_false, val_edges = val_edges, val_edges_false = val_edges_false, \
     test_edges = test_edges, test_edges_false = test_edges_false, matrix=emb_matrix_umap)
    
    lp_arg_pca = LP_arguments(emb_mappings=emb_mappings_pca, adj_sparse=adj_sparse, train_edges = train_edges, \
     train_edges_false = train_edges_false, val_edges = val_edges, val_edges_false = val_edges_false, \
     test_edges = test_edges, test_edges_false = test_edges_false, matrix=emb_matrix_pca)
    
    methods = {
        "node2vec" : lp_arg,
        "node2vec+UMAP" : lp_arg_umap,
        "node2vec+PCA": lp_arg_pca
    }
    
    adamic_adard_result = MethodResult('Adamic-Adar', aa_roc, aa_ap)
    jc_result = MethodResult('Jaccard Coefficient', jc_roc, jc_ap)
    pa_result = MethodResult('Preferential Attachment', pa_roc, pa_ap)

    methods_list = [adamic_adard_result, jc_result, pa_result]
    
    for key, value in methods.items():
        val_roc, val_ap, test_roc, test_ap = link_prediction_on_embedding(value)
        methods_list.append(MethodResult(key, test_roc, test_ap))
    
    result = LatexResults(adj_sparse.shape[0], len(train_edges), len(test_edges), methods_list)
    
    with open("results.txt", "a") as file:
        file.write(result.get_latex_representation())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--min_degree', type=int)
    args = parser.parse_args()
    args = vars(args)
    calculate(args['min_degree'])
