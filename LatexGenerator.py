class LatexResults():

    def __init__(self, number_of_nodes, training_edges, test_edges, list_of_methods_result, caption):
        self.number_of_nodes = number_of_nodes
        self.training_edges = training_edges
        self.test_edges = test_edges
        self.list_of_methods_result = list_of_methods_result
        self.caption = caption
    
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

        caption = "\caption{" + caption_txt + "}"

        rows = " ".join([self.get_row_latex_repr(method) for method in self.list_of_methods_result])
        return begining + rows + end_tabular + caption + end

    def get_row_latex_repr(self, methodResult):
        return "{} & {:.4f} & {:.4f} \\\ \\hline \n".format(methodResult.methodName, methodResult.testROC, methodResult.testPC)