# HyBench

**HyBench** is a benchmark framework for evaluating **hybrid queries** in vector databases using a MediaWiki-based dataset. It integrates structured and vector data to assess performance on real-world hybrid workloads.

## 📁 Dataset Layout

All data is located in the `data_csv_files/` directory, organized into the following subdirectories:

### `category_csv_files/`
* **category_links_clean.csv**: Contains 2 columns from MediaWiki's `category_links` table:
   * `cl_from`
   * `cl_to`

### `page_csv_files/`
* **page.csv**:
   * Columns: `page_id`, `page_title`
   * Ordered by `page_id`
* **embedding.csv**:
   * Embeddings corresponding to `page.csv` entries
   * Same row order as `page.csv`
* **page_extra.csv**:
   * Columns: `page_len`, `page_touched`, `page_namespace`

### `revision_csv_files/`
* **revision_clean.csv**:
   * Columns: `rev_id`, `rev_page`, `rev_minor_edit`, `rev_actor`, `rev_timestamp`
   * Ordered by `rev_id`

### `text_csv_files/`
* **text.csv**:
   * Columns: `old_id`, `old_text`
   * Ordered by `old_id`
* **embedding.csv**:
   * Embeddings corresponding to `text.csv`
   * Same row order as `text.csv`

## ⚙️ Benchmark Setup

Assumes the MediaWiki dataset is already prepared (except embeddings).

### Step 1: Generate Embeddings & Queries

```bash
python3 benchmark_generator.py microsoft/MiniLM-L12-H384-uncased
```

This generates:
- Page and text embeddings
- Query files for the baseline

### Step 2: Compile Required C++ Files

All C++ files need FAISS and hnswlib installed.

Sample compile command:
```bash
g++ runner.cpp -o runner -O3 -std=c++17 -fopenmp \
  -I/path/to/faiss \
  /path/to/faiss/build/faiss/libfaiss.a \
  -lopenblas -lm -fopenmp
```

Compile the following files similarly:

**index_files/:**
- `index_generation.cpp`
- `index_generation_new.cpp`
- `index_generation1.cpp`

**offsets_files/:**
- `offset_calculation.cpp`

**queries_pipelines/:**
- `binary_embeddings_creator.cpp`

Compiled output files of index_generation.cpp ,index_generation_new.cpp, index_generation1.cpp , offset_calculation.cpp and binary_embeddings_creator.cpp are index_generation, build_hnsw, index_generation1, offsets_calculation, binary_embeddings_creator respectively.

### Step 3: Generate Indexes

#### a. FAISS & HNSW Indexes (in index_files/)

Run all combinations of page/text × l2/cosine:
```bash
for t in page text; do
  for m in l2 cos; do
    ./index_generation $t $m
    ./build_hnsw $t $m
  done
done
```

#### b. Generate ID Indexes
```bash
./index_generation1 ../data_csv_files/text_csv_files/text.csv old_id_index.bin 0
./index_generation1 ../data_csv_files/page_csv_files/page.csv page_id_index.bin 0
./index_generation1 ../data_csv_files/revision_csv_files/revision_clean.csv rev_id_index.bin 0
./index_generation1 ../data_csv_files/revision_csv_files/revision_clean.csv rev_page_index.bin 1
```

### Step 4: Generate Offset Files (in offsets_files/)

```bash
./offsets_calcuation ../data_csv_files/page_csv_files/page_extra.csv page_extra_offsets.bin
./offsets_calcuation ../data_csv_files/page_csv_files/page.csv page_offsets.bin
./offsets_calcuation ../data_csv_files/revision_csv_files/revision_clean.csv revision_offsets.bin
./offsets_calcuation ../data_csv_files/text_csv_files/text.csv text_offsets.bin
./offsets_calcuation ../data_csv_files/text_csv_files/embedding.csv text_embedding_offsets.bin
```

### Step 5: Convert Embeddings to Binary (in queries_pipelines/)

```bash
./binary_embeddings_creator page
./binary_embeddings_creator text
```

## ▶️ Run the Baseline

Run all hybrid queries using both FAISS and HNSW:
```bash
cd queries_pipelines/final_Appr_queries
bash run.sh
```

## ✅ Accuracy Calculation

Run from `final_Output/`:
```bash
bash acc_final.sh <source> <library> <metric>
```

**Arguments:**
- `<source>`: `our` or `postgres`
- `<library>`: `hnswlib` or `hnsw` or `ivfflat`
- `<metric>`: `l2` or `cos`

**Example:**
```bash
bash acc_final.sh our hnswlib l2
```

## 🧰 Requirements

- FAISS (libfaiss.a) and OpenBLAS
- HNSWlib for C++
- C++17 with OpenMP
- Python 3.6+
- HuggingFace transformers (for benchmark_generator.py)
