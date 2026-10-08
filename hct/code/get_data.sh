#!/bin/sh
# Downloads the public datasets used in the article into ../data
set -e
mkdir -p ../data
curl -sSL -o ../data/pima.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
curl -sSL -o ../data/compas.csv https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
curl -sSL -o ../data/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# Breast Cancer Wisconsin (Diagnostic) is loaded from scikit-learn.
echo "data ready"
