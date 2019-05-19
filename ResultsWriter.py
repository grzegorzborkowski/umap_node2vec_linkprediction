class ModelAccuracyResults():

    def __init__(self, number_of_nodes, training_edges, test_edges, list_of_methods_result, caption, classifier):
        self.number_of_nodes = number_of_nodes
        self.training_edges = training_edges
        self.test_edges = test_edges
        self.list_of_methods_result = list_of_methods_result
        self.caption = caption
        self.classifier = classifier
    
    def get_latex_representation(self):
        begining = """
        \\begin{table}[H]
        \centering
        \\begin{tabular}{|l|c|c|}
        \\hline
        \\textbf{Name of the method} & \\textbf{ROC score} & \\textbf{Precision score} \\\ \hline
        """
       	end_tabular = """\end{tabular}""" 
        end = """
            \end{table}
        """

        caption_txt = "{} {} nodes, {} training edges and {} testing edges".format(self.caption,
            self.number_of_nodes, self.training_edges, self.test_edges)

        caption = "\caption{" + caption_txt + "," + self.classifier + " classifier}"

        rows = " ".join([self.get_row_latex_repr(method) for method in self.list_of_methods_result])
        return begining + rows + end_tabular + caption + end

    def get_row_latex_repr(self, methodResult):
        return "{} & {:.4f} & {:.4f} \\\ \\hline \n".format(methodResult.methodName, methodResult.testROC, methodResult.testPC)


    def get_csv_representation(self):
        #method_name,roc,pc
        return "".join([self.get_row_csv_repr(method) for method in self.list_of_methods_result])

    def get_row_csv_repr(self, methodResult):
        return "{},{},{},{},{},{:.4f},{:.4f}\n".format(
            methodResult.methodName, self.classifier,
            self.number_of_nodes, self.training_edges, self.test_edges,
            methodResult.testROC, methodResult.testPC)


class ModelTimeResults():

    def __init__(self, list_of_method_time, number_of_nodes, training_edges, test_edges, caption, classifier):
        self.list_of_method_time = list_of_method_time
        self.number_of_nodes = number_of_nodes
        self.training_edges = training_edges
        self.test_edges = test_edges
        self.caption = caption
        self.classifier = classifier

    def get_latex_representation(self):
        beginning = """
        \\begin{table}[H]
        \\centering
        \\begin{tabular}{|l|c|}
        \\hline
        \\textbf{Name of the method} & \\textbf{Training time (s)} \\\ \hline
        """

        end_tabular = """\end{tabular}"""
        end = """
            \end{table}
        """

        caption_txt = "{} {} nodes. {} training edges, {} test edges, {} classifier".format(
            self.caption,
            self.number_of_nodes, self.training_edges, self.test_edges,
            self.classifier)

        caption = "\caption{" + caption_txt + "}"

        rows = " ".join([self.get_row_latex_repr(method) for method in self.list_of_method_time])

        return beginning + rows + end_tabular + caption + end

    def get_row_latex_repr(self, methodTime):
        return "{} & {:.4f} \\\ \\hline \n".format(methodTime.methodName, methodTime.time)

    def get_csv_representation(self):
        #method_name,time
        return "".join([self.get_row_csv_repr(method) for method in self.list_of_method_time])

    def get_row_csv_repr(self, methodTime):
        return "{},{},{},{},{:.4f}\n".format(
            methodTime.methodName, self.classifier,
            self.number_of_nodes, self.training_edges, self.test_edges,
            methodTime.time)



