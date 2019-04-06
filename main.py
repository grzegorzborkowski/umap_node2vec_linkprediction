# Copied from: https://github.com/lucashu1/link-prediction
# @misc{lucas_hu_2018_1408472,
#   author       = {Lucas Hu and
#                   Thomas Kipf and
#                   Gökçen Eraslan},
#   title        = {{lucashu1/link-prediction: v0.1: FB and Twitter 
#                    Networks}},
#   month        = sep,
#   year         = 2018,
#   doi          = {10.5281/zenodo.1408472},
#   url          = {https://doi.org/10.5281/zenodo.1408472}
# }

import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from gensim.models import Word2Vec
import node2vec
from sklearn.linear_model import LogisticRegression
from LP_arguments import LP_arguments
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if verbose == True:
        print ('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print ('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print ("WARNING: not enough removable edges to perform full train-test split!")
        print ("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print ("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print ('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print ('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
            
        val_edges_false.add(false_edge)

    if verbose == True:
        print ('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print ('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print ('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print ('Done with train-test split!')
        print ('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
        val_edges, val_edges_false, test_edges, test_edges_false

def get_roc_score(adj_sparse, edges_pos, edges_neg, score_matrix):
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]]) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]]) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

from collections import namedtuple

MethodResult = namedtuple('MethodResult', ['methodName', 'testROC', 'testPC'])

class Results():

    def __init__(self, number_of_nodes, training_edges, test_edges, list_of_methods_result):
        self.number_of_nodes = number_of_nodes
        self.training_edges = training_edges
        self.test_edges = test_edges
        self.list_of_methods_result = list_of_methods_result
    
    def get_latex_representation(self):
        begining = """
        \\begin{table}[]
        \\begin{tabular}{|l|l|l|}
        \\hline
        \\textbf{Name of the method} & \\textbf{ROC score} & \\textbf{AP score} \\\ \hline
        """
        
        end = """
            \end{tabular}
            \end{table}
        """

        rows = " ".join([self.get_row_latex_repr(method) for method in self.list_of_methods_result])
        return begining + rows + end

    def get_row_latex_repr(self, methodResult):
        return "{} & {} & {} \\\ \\hline \n".format(methodResult.methodName, methodResult.testROC, methodResult.testPC)

def calculate(file_path="graph.graph"):
    graph = nx.read_edgelist(file_path, delimiter=" ")
    min_degree = 200
    nodes = [node for node, degree in graph.degree().items() if degree >= min_degree]
    graph = graph.subgraph(nodes)
    connected_components = nx.connected_components(graph)
    largest_cc_nodes = max(connected_components, key=len)
    graph = graph.subgraph(largest_cc_nodes)
    print (graph)
    adj_sparse = nx.to_scipy_sparse_matrix(graph)
    adj = nx.to_numpy_matrix(graph)
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.1)

    # Inspect train/test split
    print ("Total nodes:", adj_sparse.shape[0])
    print ("Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
    print ("Training edges (positive):", len(train_edges))
    print ("Training edges (negative):", len(train_edges_false))
    print ("Validation edges (positive):", len(val_edges))
    print ("Validation edges (negative):", len(val_edges_false))
    print ("Test edges (positive):", len(test_edges))
    print ("Test edges (negative):", len(test_edges_false))

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
    print (emb_mappings)
    
    umap_obj = umap.UMAP(n_neighbors=10, min_dist=0.03, n_components=24)
    emb_mappings_umap = umap_obj.fit_transform(emb_matrix)
    print (emb_mappings_umap)

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
    
    result = Results(adj_sparse.shape[0], len(train_edges), len(test_edges), methods_list)
    print(result.get_latex_representation())


def link_prediction_on_embedding(lp_arg):
    emb_mappings = lp_arg.emb_mappings
    adj_sparse = lp_arg.adj_sparse
    train_edges = lp_arg.train_edges
    train_edges_false = lp_arg.train_edges_false
    val_edges = lp_arg.val_edges
    val_edges_false = lp_arg.val_edges_false
    test_edges = lp_arg.test_edges
    test_edges_false = lp_arg.test_edges_false
    emb_matrix = lp_arg.matrix

    # Train-set edge embeddings
    pos_train_edge_embs = get_edge_embeddings(train_edges,emb_matrix)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false,emb_matrix)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    # Val-set edge embeddings, labels
    pos_val_edge_embs = get_edge_embeddings(val_edges,emb_matrix)
    neg_val_edge_embs = get_edge_embeddings(val_edges_false,emb_matrix)
    val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
    val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges,emb_matrix)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false,emb_matrix)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    val_roc = roc_auc_score(val_edge_labels, val_preds)
    val_ap = average_precision_score(val_edge_labels, val_preds)

    # Predicted edge scores: probability of being of class "1" (real edge)
    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    test_roc = roc_auc_score(test_edge_labels, test_preds)
    test_ap = average_precision_score(test_edge_labels, test_preds)

    return val_roc, val_ap, test_roc, test_ap


    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
def get_edge_embeddings(edge_list,emb_matrix):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        emb2 = emb_matrix[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs
    

if __name__ == "__main__":
    calculate()
