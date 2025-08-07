# Step 1: Generate benchmark data using MiniLM 
# You can use other embedding model by simply replacing the first argument to any one present in the models_supported.txt file in the database-generation directory
python3 benchmark_generator.py microsoft/MiniLM-L12-H384-uncased

# Step 2: Compile C++ index generators
# You need to put the real path of faiss in the first compile command depending on the running system
g++ index_files/ivfflat_index_generator.cpp -o index_files/ivfflat_index_generator -O3 -std=c++17 -I/path/to/faiss /path/to/faiss/build/faiss/libfaiss.a -lopenblas -lm -fopenmp
g++ index_files/hnsw_index_generator.cpp -o index_files/hnsw_index_generator -O3 -std=c++17 -lopenblas -lm -fopenmp
g++ index_files/hash_table_based_index_generator.cpp -o index_files/hash_table_based_index_generator -O3 -std=c++17
g++ offsets_files/offset_calculation.cpp -o offsets_files/offset_calculation -O3 -std=c++17
g++ binary_embeddings_creator.cpp -o binary_embeddings_creator -O3 -std=c++17

# Step 3: Create indexes for IVF and HNSW for page and text
for t in page text; do
  for m in l2 cos; do
    ./index_files/ivfflat_index_generator $t $m
    ./index_files/hnsw_index_generator $t $m
  done
done

# Step 4: Create hash-based indexes
./index_files/hash_table_based_index_generator ../data_csv_files/text_csv_files/text.csv old_id_index.bin 0
./index_files/hash_table_based_index_generator ../data_csv_files/page_csv_files/page.csv page_id_index.bin 0
./index_files/hash_table_based_index_generator ../data_csv_files/revision_csv_files/revision_clean.csv rev_id_index.bin 0
./index_files/hash_table_based_index_generator ../data_csv_files/revision_csv_files/revision_clean.csv rev_page_index.bin 1

# Step 5: Calculate offsets
./offsets_files/offset_calculation ../data_csv_files/page_csv_files/page_extra.csv page_extra_offsets.bin
./offsets_files/offset_calculation ../data_csv_files/page_csv_files/page.csv page_offsets.bin
./offsets_files/offset_calculation ../data_csv_files/revision_csv_files/revision_clean.csv revision_offsets.bin
./offsets_files/offset_calculation ../data_csv_files/text_csv_files/text.csv text_offsets.bin
./offsets_files/offset_calculation ../data_csv_files/text_csv_files/embedding.csv text_embedding_offsets.bin

# Step 6: Create binary embeddings
./binary_embeddings_creator page
./binary_embeddings_creator text
