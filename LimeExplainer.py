import lime.lime_tabular
import tqdm
import matplotlib.pyplot as plt
import numpy as np

class LimeExplainerPlotter():

    def __init__(self, method_name, id_to_importance_dict, id_to_occurs_in_top_5, importance_sum):
        self.method_name = method_name
        self.id_to_importance_dict = id_to_importance_dict
        self.importance_sum = importance_sum
        self.id_to_occurs_in_top_5 = id_to_occurs_in_top_5

    def plot_feature_importance(self):
        x_values = []
        for feature, importance in sorted(self.id_to_importance_dict.items(), key=lambda p:p[1], reverse=True):
            print(str(feature)+': '+str((importance/self.importance_sum)*100)+', in top 5: '+str(self.id_to_occurs_in_top_5[feature]))
            x_values.append((importance/self.importance_sum)*100)
        plt.title(self.method_name)
        plt.bar(np.arange(len(x_values)), x_values)
        plt.show()

class LimeExplainer():

    def __init__(self, method_name, edge_classifier, train_edge_embs, train_edge_labels, test_edge_embs):
        self.method_name = method_name
        self.edge_classifier = edge_classifier
        self.train_edge_embs = train_edge_embs
        self.train_edge_labels = train_edge_labels
        self.test_edge_embs = test_edge_embs
        self.__create_explainer_object__()
        self.id_to_importance_dict = {}
        self.id_to_occurs_in_top_5 = {}
        for i in range(len(self.test_edge_embs[0])):
            self.id_to_importance_dict[i] = 0
            self.id_to_occurs_in_top_5[i] = 0

    def __create_explainer_object__(self):
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.train_edge_embs,training_labels=self.train_edge_labels)

    def get_explanations(self):
        print ("----------------------------------------")
        print (self.method_name)
        print ("----------------------------------------")
        importance_sum = 0
        for emb in tqdm.tqdm(self.test_edge_embs[:100]):
            exp = self.explainer.explain_instance(emb, self.edge_classifier.predict_proba)
            exps = exp.as_list()

            for i in range(len(exps)):
                feature_exp = exps[i]
                feature_equation = feature_exp[0]
                importance = abs(feature_exp[1])
                if len(feature_equation.split(' ')) == 3: # "5 <= 0.99"
                    feature = int(feature_equation.split(' ')[0])
                else: # "0.75 < 5 < 5.99"
                    feature = int(feature_equation.split(' ')[2])
                self.id_to_importance_dict[feature] += importance
                if i < 5:
                    self.id_to_occurs_in_top_5[feature] += 1
                importance_sum += importance

        explainer = LimeExplainerPlotter(self.method_name, self.id_to_importance_dict, self.id_to_occurs_in_top_5, importance_sum)
        explainer.plot_feature_importance()

