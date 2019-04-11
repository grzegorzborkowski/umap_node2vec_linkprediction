import networkx as nx
import scipy.sparse as sp
import numpy as np
from gensim.models import Word2Vec
import node2vec
from LP_arguments import LP_arguments
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
import argparse
from link_prediction_helpers import *
from collections import namedtuple

MethodResult = namedtuple('MethodResult', ['methodName', 'testROC', 'testPC'])



def calculate(min_degree, file_path="graph.graph"):
    graph = nx.read_edgelist(file_path, delimiter=" ")
    nodes = [node for node, degree in graph.degree().items() if degree >= min_degree]
    graph = graph.subgraph(nodes)
    connected_components = nx.connected_components(graph)
    largest_cc_nodes = max(connected_components, key=len)
    graph = graph.subgraph(largest_cc_nodes)
    # print (graph)
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

    P = 1 # Return hyperparameter
    Q = 1 # In-out hyperparameter
    WINDOW_SIZE = 12 # Context size for optimization
    NUM_WALKS = 20 # Number of walks per source
    WALK_LENGTH = 5 # Length of walk per source
    DIMENSIONS = 32 # Embedding dimension
    DIRECTED = False # Graph directed/undirected
    WORKERS = 8 # Num. parallel workers
    ITER = 10 # SGD epochs

    # Preprocessing, generate walks
    g_n2v = node2vec.Graph(g_train, DIRECTED, P, Q) # create node2vec graph instance
    g_n2v.preprocess_transition_probs()
    walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
    walks = [map(str, walk) for walk in walks]

    # Train skip-gram model
    walks = [list(map(str, walk)) for walk in walks] # convert each vertex id to a string
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

    # Store embeddings mapping
    emb_mappings = model.wv
    emb_list = []
    for node_index in range(0, adj_sparse.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)
    
    umap_obj = umap.UMAP(n_neighbors=10, min_dist=0.03, n_components=24)
    emb_mappings_umap = umap_obj.fit_transform(emb_matrix)
    
    pca_obj = PCA(n_components=24)
    emb_mappings_pca = pca_obj.fit_transform(emb_matrix)
 
    lp_arg = LP_arguments(emb_mappings=emb_mappings, adj_sparse = adj_sparse, train_edges = train_edges, \
     train_edges_false = train_edges_false, val_edges = val_edges, val_edges_false = val_edges_false, \
     test_edges = test_edges, test_edges_false = test_edges_false, matrix=emb_matrix)
    
    lp_arg_umap = LP_arguments(emb_mappings=emb_mappings_umap, adj_sparse=adj_sparse, train_edges = train_edges, \
     train_edges_false = train_edges_false, val_edges = val_edges, val_edges_false = val_edges_false, \
     test_edges = test_edges, test_edges_false = test_edges_false, matrix=emb_mappings_umap)
    
    lp_arg_pca = LP_arguments(emb_mappings=emb_mappings_pca, adj_sparse=adj_sparse, train_edges = train_edges, \
     train_edges_false = train_edges_false, val_edges = val_edges, val_edges_false = val_edges_false, \
     test_edges = test_edges, test_edges_false = test_edges_false, matrix=emb_mappings_pca)
    
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
