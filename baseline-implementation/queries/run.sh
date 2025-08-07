#  You need to put the actual path to faiss in the running system for the below compile command to work
 g++ runner.cpp -o runner -O3 -std=c++17 -I/path/to/faiss /path/to/faiss/build/faiss/libfaiss.a -lopenblas -lm -fopenmp

echo "Running ./runner with input for hnswlib l2 page..."
./runner hnswlib l2 page <<EOF
$(for i in {1..18}; do
    echo "1 50"
    echo "1 50"
    echo "1 50"
    echo "2 100"
    echo "2 100"
    echo "2 100"
    echo "3 200"
    echo "3 200"
    echo "3 200"
    echo "done"
done)
EOF

echo "Running ./runner with input for hnswlib l2 text..."
./runner hnswlib l2 text <<EOF
$(for i in {1..18}; do
    echo "1 50"
    echo "1 50"
    echo "1 50"
    echo "2 100"
    echo "2 100"
    echo "2 100"
    echo "3 200"
    echo "3 200"
    echo "3 200"
    echo "done"
done)
EOF

echo "Running ./runner with input for hnswlib l2 te..."
./runner hnswlib l2 te <<EOF
$(for i in {1..18}; do
    echo "1"
    echo "1"
    echo "1"
    echo "2"
    echo "2"
    echo "2"
    echo "3"
    echo "3"
    echo "3"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat l2 page..."
./runner faiss_ivfflat l2 page <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat l2 text..."
./runner faiss_ivfflat l2 text <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat cos page..."
./runner faiss_ivfflat cos page <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat cos text..."
./runner faiss_ivfflat cos text <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF

echo "Running ./runner with input for hnswlib cos text..."
./runner hnswlib cos text <<EOF
$(for i in {1..18}; do
    echo "1 50"
    echo "1 50"
    echo "1 50"
    echo "2 100"
    echo "2 100"
    echo "2 100"
    echo "3 200"
    echo "3 200"
    echo "3 200"
    echo "done"
done)
EOF

echo "Running ./runner with input for hnswlib cos page..."
./runner hnswlib cos page <<EOF
$(for i in {1..18}; do
    echo "1 50"
    echo "1 50"
    echo "1 50"
    echo "2 100"
    echo "2 100"
    echo "2 100"
    echo "3 200"
    echo "3 200"
    echo "3 200"
    echo "done"
done)
EOF

echo "Running ./runner with input for hnswlib cos te..."
./runner hnswlib cos te <<EOF
$(for i in {1..18}; do
    echo "1"
    echo "1"
    echo "1"
    echo "2"
    echo "2"
    echo "2"
    echo "3"
    echo "3"
    echo "3"
    echo "done"
done)
EOF

echo "Running ./runner with input for hnswlib cos pe..."
./runner hnswlib cos pe <<EOF
$(for i in {1..18}; do
    echo "1"
    echo "1"
    echo "1"
    echo "2"
    echo "2"
    echo "2"
    echo "3"
    echo "3"
    echo "3"
    echo "done"
done)
EOF

echo "Running ./runner with input for hnswlib cos te..."
./runner hnswlib l2 pe <<EOF
$(for i in {1..18}; do
    echo "1"
    echo "1"
    echo "1"
    echo "2"
    echo "2"
    echo "2"
    echo "3"
    echo "3"
    echo "3"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat cos text..."
./runner faiss_ivfflat cos pe <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat cos text..."
./runner faiss_ivfflat l2 pe <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat l2 te..."
./runner faiss_ivfflat l2 te <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF

echo "Running ./runner with input for faiss_ivfflat cos te..."
./runner faiss_ivfflat cos te <<EOF
$(for i in {1..18}; do
    echo "1 5"
    echo "1 5"
    echo "1 5"
    echo "2 5"
    echo "2 5"
    echo "2 5"
    echo "3 5"
    echo "3 5"
    echo "3 5"
    echo "done"
done)
EOF
