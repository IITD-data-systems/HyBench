---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started

This guide walks through setup, database generation, benchmark execution, and accuracy evaluation.

## Requirements

Install the core system dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev libomp-dev python3 python3-venv python3-pip
```

RVBench also requires:

- C++17 compiler with OpenMP support
- FAISS, including `libfaiss.a`
- HNSWlib for C++
- Python 3.8+ recommended
- Hugging Face `transformers`
- PostgreSQL with pgvector, only if recomputing ground truth

{: .important }
> Update FAISS/OpenBLAS include and library paths in the shell scripts before compiling.

## 1. Clone RVBench

```bash
git clone https://github.com/IITD-data-systems/RVBench.git
cd RVBench
```

## 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

If the repository includes `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

Otherwise install the common packages manually:

```bash
pip install numpy pandas torch transformers sentence-transformers psycopg2-binary openpyxl
```

## 3. Prepare CSV inputs

Place the MediaWiki-derived CSV files under:

```text
database-generation/data_csv_files/
├── category_csv_files/
│   └── category_links_clean.csv
├── page_csv_files/
│   ├── page.csv
│   ├── page_extra.csv
│   └── embedding.csv
├── revision_csv_files/
│   └── revision_clean.csv
└── text_csv_files/
    ├── text.csv
    └── embedding.csv
```

The embedding files must preserve the same row order as their source CSV files.

## 4. Generate database artifacts and queries

```bash
cd database-generation
bash database_and_query_generator.sh
```

The script generates embeddings, query files, index files, and offset files.

## 5. Run the benchmark

```bash
cd ../baseline-implementation/queries
bash run.sh
```

## 6. Evaluate accuracy

```bash
cd ../../output-files
bash accuracy_baseline.sh
```

## Sampling a smaller dataset

```bash
cd database-generation
python3 sampling.py <rows_in_page_table>
```

Example:

```bash
python3 sampling.py 500000
```

The requested number of rows must be smaller than the original `page.csv` row count.

## Recompute ground truth

Ground truth is precomputed for the default experimental setup. Recompute it if you change the dataset, embeddings, metrics, or query parameters.

```bash
cd output-files
python3 ground_truth_result_computer.py
```
