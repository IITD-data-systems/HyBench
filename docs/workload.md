---
layout: default
title: Benchmark Workload
nav_order: 3
---

# Benchmark Workload

RVBench contains 39 parameterized hybrid query templates. The workload is organized by similarity semantics and SQL complexity.

## Similarity semantics

### Nearest-neighbor queries

Nearest-neighbor queries retrieve the most similar items using top-k or distance-threshold constraints.

Examples:

- Top-k pages most similar to a query vector.
- Similar pages filtered by page length.
- Similar revisions joined with timestamp predicates.
- Similar results grouped by revision author or year.

### Interval-neighbor queries

Interval-neighbor queries retrieve items whose similarity rank or distance falls within a specified interval.

Examples:

- Pages ranked between positions `[l, u]`.
- Pages within a distance band `[d_min, d_max]`.
- Interval results combined with joins, filters, and aggregation.

### Sampled-neighbor queries

Sampled-neighbor queries retrieve items at selected similarity ranks or across multiple distance ranges.

Examples:

- Pages at ranks `[1, 3, 5, 7]`.
- Pages from several distance bands.
- Sampled neighbors joined with revision metadata.

## SQL constructs

RVBench combines vector similarity with SQL constructs at three levels.

| Level | SQL constructs | Purpose |
|---|---|---|
| Basic | projection, filters | Test vector index quality and predicate selectivity. |
| Intermediate | joins, group-by, aggregation | Test relational-vector execution strategies. |
| Advanced | CTEs, subqueries, CASE, set operations | Test complex planner behavior and composability. |

## Metrics

RVBench measures both performance and quality.

| Metric | Use |
|---|---|
| Query latency | Measures execution time. |
| Precision | Measures result overlap for non-aggregate queries. |
| RMSE | Measures aggregate-result error for grouped or aggregate queries. |

## Output organization

```text
output-files/
├── brute_queries_output/
├── baseline_queries_output/
├── pgvector_query_plans/
├── postgres_queries_output/
├── experiments/
├── ground_truth_result_computer.py
└── accuracy_baseline.sh
```
