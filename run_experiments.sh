rm results.txt

for min_degree in 25 50 100 150
do
	echo "Calling program with min_degree: $min_degree"
	python3 main.py --min_degree $min_degree

done
