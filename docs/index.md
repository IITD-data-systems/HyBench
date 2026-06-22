---
layout: default
title: Home
nav_order: 1
description: "RVBench is a benchmark for hybrid relational-vector database systems."
permalink: /
---

# RVBench
{: .fs-9 }

Benchmarking hybrid relational-vector database systems.
{: .fs-6 .fw-300 }

[Get started](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View workload](workload){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[GitHub](https://github.com/IITD-data-systems/RVBench){: .btn .fs-5 .mb-4 .mb-md-0 }

---

![RVBench architecture](assets/images/rvbench-architecture.svg)

## What RVBench evaluates

RVBench evaluates databases that combine **relational operators** with **vector similarity search** in the same analytical workload. It is designed for database and vector-search researchers who want to test planner behavior, index selection, latency, and retrieval quality for realistic hybrid workloads.

{: .highlight }
> RVBench goes beyond simple filter-plus-vector-search queries. It includes top-k nearest-neighbor workloads, rank-interval workloads, and sampled-neighbor workloads, combined with SQL constructs such as joins, filters, grouping, aggregation, CTEs, subqueries, and set operations.

## Why it matters

AI-powered systems often need both semantic search and structured reasoning. A user may ask for pages similar to a query, but only within a date range, category, author group, or metadata predicate. RVBench makes these patterns concrete and reproducible.

## Key features

| Capability | Description |
|---|---|
| MediaWiki-derived schema | Uses Page, Text, Revision, and CategoryLinks tables with vector embeddings. |
| 39 query templates | Covers nearest-neighbor, interval-neighbor, and sampled-neighbor workloads. |
| Pluggable embeddings | Supports transformer-based embedding generation and custom model integration. |
| Scalable datasets | Supports sampling and scale-factor based evaluation. |
| Reference implementation | Includes C++17 baseline execution with vector index backends. |
| Accuracy tooling | Measures latency plus quality metrics such as precision and RMSE. |

## Quick commands

```bash
git clone https://github.com/IITD-data-systems/RVBench.git
cd RVBench

cd database-generation
bash database_and_query_generator.sh

cd ../baseline-implementation/queries
bash run.sh

cd ../../output-files
bash accuracy_baseline.sh
```

## Recommended next steps

1. Read the [getting started guide](getting-started).
2. Review the [workload taxonomy](workload).
3. Add the [citation](citation) to your paper if RVBench helps your research.
