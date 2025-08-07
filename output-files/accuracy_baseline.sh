#!/bin/bash

# Run accuracy calculation for the baseline implementation using hnswlib index with L2 distance
bash accuracy_calculator.sh baseline hnswlib l2

# Run accuracy calculation for the baseline implementation using hnswlib index with Cosine distance
bash accuracy_calculator.sh baseline hnswlib cos

# Run accuracy calculation for the baseline implementation using ivfflat index with L2 distance
bash accuracy_calculator.sh baseline ivfflat l2

# Run accuracy calculation for the baseline implementation using ivfflat index with Cosine distance
bash accuracy_calculator.sh baseline ivfflat cos
