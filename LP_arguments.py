class LP_arguments():
    def __init__(self, emb_mappings, adj_sparse, train_edges, train_edges_false, \
     val_edges, val_edges_false, test_edges, test_edges_false,matrix):
        self.emb_mappings = emb_mappings
        self.adj_sparse = adj_sparse
        self.train_edges = train_edges
        self.train_edges_false = train_edges_false
        self.val_edges = val_edges
        self.val_edges_false = val_edges_false
        self.test_edges = test_edges
        self.test_edges_false = test_edges_false
        self.matrix = matrix