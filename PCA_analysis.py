class PCA_analysis():
    
    def __init__(self, pca_obj):
        self.pca_obj = pca_obj

    def print_analysis(self):
        print (self.pca_obj.explained_variance_ratio_)

