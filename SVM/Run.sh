#!/bin/bash

echo "running Q2 ..."

python Q2-SVM-Primal.py

echo "running Q3 Parts a,b,c..."
python Q3   a,b,c (SVM-Kernel).py

echo "running Q3 part (d) (kernel Perceptron)..."
python Q3 -d (kernel Perceptron).py