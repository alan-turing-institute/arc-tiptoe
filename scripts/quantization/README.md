# Quantization

Experiments on effects of quantization

## create_non_quantized_clusters.py

**Purpose**: Creates a `non_quantized_clusters` dir that mirrors `clusters` but without quantization.

**Description**: 
- Loads preprocessed data
- Creates non-quantized cluster equivalents
- Saves to data dir in new subdir `<data_dir>/non_quantized_clusters`

**Usage**:
```bash


# Specify dataset and parameters
python scripts/quantization/create_non_quantized_clusters.py \
    --data_dir /path/to/dir
