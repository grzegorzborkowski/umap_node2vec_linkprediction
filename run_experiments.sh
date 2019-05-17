rm results.txt

for min_degree in 10 30 50 70
do
    for dataset in graph.graph external_graph.csv
    do
	echo "Calling program with min_degree: $min_degree, dataset: $dataset"
	python3 main.py --min_degree $min_degree --dataset_path $dataset --analyse yes --classifier SVM
    done
done
