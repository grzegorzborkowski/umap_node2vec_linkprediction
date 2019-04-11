import abc
import json
import node2vec
from gensim.models import Word2Vec

class AbstractModel(abc.ABC):

    def get_config_filename(self):
        return "config.json"

    @abc.abstractmethod
    def get_model(self):
        pass

class Node2vecModel(AbstractModel):

    def get_model(self, g_train):
        params = {}
        with open(self.get_config_filename(), 'r') as f:
            datastore = json.load(f)
            params = datastore['node2vec']
            directed = params['DIRECTED'] == "true"
            g_n2v = node2vec.Graph(g_train, directed, params['P'], params['Q'])
            g_n2v.preprocess_transition_probs()
            walks = g_n2v.simulate_walks(params['NUM_WALKS'], params['WALK_LENGTH'])
            walks = [map(str, walk) for walk in walks]
            walks = [list(map(str, walk)) for walk in walks] # convert each vertex id to a string
            model = Word2Vec(walks, size=params['DIMENSIONS'], window=params['WINDOW_SIZE'], \
            min_count=0, sg=1, workers=params['WORKERS'], iter=params['ITER'])
            return model

class ModelFactory:

    def get_model(self, model_name, g_train):
        if model_name == "node2vec":
            return Node2vecModel().get_model(g_train)
        else:
            raise ("Unknown model. Can't create")