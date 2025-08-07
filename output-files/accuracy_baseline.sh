#!/bin/bash

echo "Running: bash accuracy_calculator.sh baseline hnswlib l2"
bash accuracy_calculator.sh baseline hnswlib l2

echo "Running: bash accuracy_calculator.sh baseline hnswlib cos"
bash accuracy_calculator.sh baseline hnswlib cos

echo "Running: bash accuracy_calculator.sh baseline ivfflat l2"
bash accuracy_calculator.sh baseline ivfflat l2

echo "Running: bash accuracy_calculator.sh baseline ivfflat cos"
bash accuracy_calculator.sh baseline ivfflat cos
