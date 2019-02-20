#!/bin/bash

for clf in svm lstm; do
	for vectorSource in custom-fasttext fasttext none; do
		filename="$clf-$vectorSource"
		echo "PYTHONHASHSEED=0 ; pipenv run python ./$clf-20news.py $vectorSource > $filename.out"
		PYTHONHASHSEED=0 ; pipenv run python ./$clf-20news.py $vectorSource > "$filename.out" 
	done
done

