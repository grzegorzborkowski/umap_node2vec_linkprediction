import abc
import json
import node2vec
from gensim.models import Word2Vec
import umap
from sklearn.decomposition import PCA

class AbstractModel(abc.ABC):

    def get_config_filename(self):
        return "config.json"

    @abc.abstractmethod
    def get_model(self):
        pass

class Node2vecModel(AbstractModel):

    def get_model(self, name, g_train):
        params = {}
        with open(self.get_config_filename(), 'r') as f:
            datastore = json.load(f)
            if name == "node2vec_32":
                params = datastore['node2vec_32']
            else:
                params = datastore['node2vec_16']
            directed = params['DIRECTED'] == "true"
            g_n2v = node2vec.Graph(g_train, directed, params['P'], params['Q'])
            g_n2v.preprocess_transition_probs()
            walks = g_n2v.simulate_walks(params['NUM_WALKS'], params['WALK_LENGTH'])
            walks = [map(str, walk) for walk in walks]
            walks = [list(map(str, walk)) for walk in walks] # convert each vertex id to a string
            model = Word2Vec(walks, size=params['DIMENSIONS'], window=params['WINDOW_SIZE'], \
            min_count=0, sg=1, workers=params['WORKERS'], iter=params['ITER'])
            return model

class UMAPModel(AbstractModel):

    def get_model(self):
        with open(self.get_config_filename(), 'r') as f:
            datastore = json.load(f)['UMAP_16']
            umap_obj = umap.UMAP(n_neighbors=datastore['n_neighbours'], \
                min_dist=datastore['min_dist'], n_components=datastore['n_components'])
            return umap_obj

class PCAModel(AbstractModel):

    def get_model(self):
        with open(self.get_config_filename(), 'r') as f:
            components = json.load(f)['PCA_16']['n_components']
            return PCA(n_components=components)

class ModelFactory:

    def __init__(self, g_train):
        self.g_train = g_train

    def get_model(self, model_name):
        if model_name == "node2vec_32":
            return Node2vecModel().get_model(model_name, self.g_train)
        if model_name == "node2vec_16":
            return Node2vecModel().get_model(model_name, self.g_train)
        if model_name == "UMAP_16":
            return UMAPModel().get_model()
        if model_name == "PCA_16":
            return PCAModel().get_model()
        else:
            raise ("Unknown model. Can't create")