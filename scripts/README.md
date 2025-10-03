# Scripts

## Overview

The scripts in this directory provide analysis and evaluation tools for:
- Evaluating TF-IDF baseline search performance
- Analyzing search results from CSV files (clustering retrieval results)
- Preprocessing configurations

## Scripts

### `eval-tf-idf.py`

**Purpose**: Evaluates TF-IDF based information retrieval performance on IR datasets.

**Description**: This script provides a comprehensive TF-IDF baseline evaluation system that:
- Loads documents and queries from IR datasets
- Trains or loads existing TF-IDF models
- Performs batch search operations
- Evaluates search quality using standard IR metrics

**Usage**:
```bash
# Basic usage with defaults
python scripts/eval-tf-idf.py

# Specify dataset and parameters
python scripts/eval-tf-idf.py \
    --dataset_name "msmarco-document/trec-dl-2020" \
    --max_documents 1000 \
    --num_results 50 \
    --log_level INFO

# Quick test with small dataset
python scripts/eval-tf-idf.py \
    --max_documents 100 \
    --batch_size 50 \
    --num_results 10 \
    --log_level DEBUG
```

**Configuration Options**:
- `--dataset_name`: IR dataset name (from ir_datasets library)
- `--max_documents`: Maximum documents to process (None for all)
- `--num_results`: Number of search results to return per query
- `--batch_size`: Document processing batch size
- `--log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Output**:
- Search results JSON files in `results/{dataset_name}/tfidf/`
- Mean evaluation metrics in `results/{dataset_name}/tfidf/mean_metrics/`

---

### `analysis_from_csv.py`

**Purpose**: Analyzes clustering-based search results from CSV files.

**Description**: This script processes search results that have been saved to CSV format, typically from clustering experiments. It:
- Parses search results from CSV files
- Loads corresponding IR dataset queries
- Evaluates search performance across different cluster configurations
- Saves detailed metrics for each clustering configuration

**Usage**:
```bash
# Basic usage
python scripts/analysis_from_csv.py

# Specify custom CSV file and parameters
python scripts/analysis_from_csv.py \
    --csv_filepath "results/msmarco-document_trec-dl-2019/distilbert/search_results/experiment.csv" \
    --n_results 50 \
    --log_level INFO
```

**Configuration Options**:
- `--csv_filepath`: Path to CSV file containing search results
- `--n_results`: Number of top results to consider for evaluation
- `--log_level`: Logging verbosity level

**Expected CSV Format**: The script expects CSV files with clustering search results where:
- Results are organized by number of clusters
- Each row represents a query-document pair with relevance scores
- Dataset name is inferred from the file path structure

**Output**:
- Evaluation metrics saved to `results/{dataset_name}/embedding_model/mean_metrics/`
- Separate JSON files for each cluster configuration

---

### 3. `preprocess_from_config.py`

(TODO)
---

## Common Patterns

### Dataset Support
Both scripts work with IR datasets through the `ir_datasets` library, specifically supporting:
- MS MARCO variants
- WIKIR variants

### Output Structure
Results are saved in a consistent directory structure:
```
results/
├── {dataset_name}/
│   ├── tfidf/                     # TF-IDF results
│   │   ├── search_results/
│   │   │   └── {n_docs}_docs.json
│   │   └── mean_metrics/
│   │       └── {n_docs}_docs/
│   │           └── {n_results}_results.json
│   │
│   └── embedding_model/           # Clustering results
│       └── mean_metrics/
│           └── {n_results}_results/
│               └── {n_clusters}_clusters.json
```

### Evaluation Metrics
Both scripts compute standard information retrieval metrics:
- **Precision@K**: Precision at top K results
- **Recall@K**: Recall at top K results
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
