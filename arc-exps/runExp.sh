# Run local Tiptoe performance experiments

set -e

# Config
PREAMBLE="/home/azureuser"
OUTPUT_DIR="/home/azureuser/experiment_results"
NUM_EMBED_SERVERS=4
NUM_URL_SERVERS=1

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=== Tiptoe Local Performance Experiments ==="
echo "Preamble: $PREAMBLE"
echo "Output Directory: $OUTPUT_DIR"
echo "Embedding Servers: $NUM_EMBED_SERVERS"
echo "URL Servers: $NUM_URL_SERVERS"
echo ""

# Function to clean up any existing processes
cleanup_processes() {
    echo "Cleaning up existing processes..."
    pkill -f "go run.*server" 2>/dev/null || true
    pkill -f "go run.*coordinator" 2>/dev/null || true
    sleep 5
}

# Function to check if data exists
check_data() {
    echo "Checking data availability..."
    
    # Check cluster data
    if [ ! -d "$PREAMBLE/data/clusters" ]; then
        echo "ERROR: Cluster data not found at $PREAMBLE/data/clusters"
        exit 1
    fi
    
    # Check FAISS index
    if [ ! -f "$PREAMBLE/data/artifact/dim192/index.faiss" ]; then
        echo "ERROR: FAISS index not found at $PREAMBLE/data/artifact/dim192/index.faiss"
        exit 1
    fi
    
    # Check PCA components for embeddings
    if [ ! -f "$PREAMBLE/data/embeddings/pca_components_192.txt" ]; then
        echo "WARNING: PCA components not found at $PREAMBLE/data/embeddings/pca_components_192.txt"
    fi
    
    # Check cluster centroids
    if [ ! -f "$PREAMBLE/data/embeddings/centroids.txt" ]; then
        echo "WARNING: Cluster centroids not found at $PREAMBLE/data/embeddings/centroids.txt"
    fi
    
    # Check MSMARCO queries
    if [ ! -f "$PREAMBLE/msmarco_data/msmarco-docdev-queries.tsv" ]; then
        echo "WARNING: MSMARCO queries not found at $PREAMBLE/msmarco_data/msmarco-docdev-queries.tsv"
    fi
    
    echo "✓ Basic data checks passed"
}

# Function to run baseline experiments
run_baseline_experiments() {
    echo "=== Running Baseline Experiments ==="
    
    cd arc-exps
    
    # Run your refactored experiment script (FIXED: removed --output_dir argument)
    echo "Starting baseline performance experiments..."
    python3 runExp_refactor.py \
        --num_embed_servers $NUM_EMBED_SERVERS \
        --num_url_servers $NUM_URL_SERVERS \
        --preamble $PREAMBLE 2>&1 | tee $OUTPUT_DIR/baseline_experiment.log
    
    cd ..
    echo "✓ Baseline experiments completed"
}

# Function to run quality evaluation
run_quality_evaluation() {
    echo "=== Running Quality Evaluation ==="
    
    # Install faiss if not available
    if ! python3 -c "import faiss" 2>/dev/null; then
        echo "Installing faiss-cpu..."
        pip3 install faiss-cpu
    fi
    
    # Check if sentence-transformers is available
    if ! python3 -c "import sentence_transformers" 2>/dev/null; then
        echo "Installing sentence-transformers..."
        pip3 install sentence-transformers
    fi
    
    # Use clustering
    cd clustering
    
    # Create config file for your setup
    cat > azure_config.json << EOF
{
    "pca_components_file": "$PREAMBLE/data/embeddings/pca_components_192.txt",
    "query_file": "$PREAMBLE/msmarco_data/msmarco-docdev-queries.tsv",
    "cluster_file_location": "$PREAMBLE/data/clusters/",
    "url_bundle_base_dir": "$PREAMBLE/data/clusters/",
    "is_text": true,
    "run_pca": true,
    "run_url_filter": false,
    "url_filter_by_cluster": false,
    "run_msmarco_dev_queries": true,
    "filter_badwords": false,
    "index_file": "$PREAMBLE/data/artifact/dim192/index.faiss",
    "short_exp": true
}
EOF

    echo "Running search quality evaluation..."
    python3 search.py azure_config.json > $OUTPUT_DIR/quality_evaluation.log 2>&1
    
    cd ../
    echo "✓ Quality evaluation completed"
}

# Function to analyze results
analyze_results() {
    echo "=== Analyzing Results ==="
    
    # Create summary script
    cat > $OUTPUT_DIR/analyze_results.py << 'EOF'
#!/usr/bin/env python3
import os
import json
import glob

def analyze_experiment_logs(output_dir):
    print("=== Experiment Results Summary ===")
    
    # Check baseline experiment logs
    baseline_log = f"{output_dir}/baseline_experiment.log"
    if os.path.exists(baseline_log):
        print(f"\n📊 Baseline Experiment Log:")
        print(f"   Location: {baseline_log}")
        
        # Extract key metrics (you'll need to adapt based on your log format)
        with open(baseline_log, 'r') as f:
            content = f.read()
            if "latency" in content.lower():
                print("   ✓ Contains latency data")
            if "throughput" in content.lower():
                print("   ✓ Contains throughput data")
    
    # Check quality evaluation
    quality_log = f"{output_dir}/quality_evaluation.log"
    if os.path.exists(quality_log):
        print(f"\n🔍 Quality Evaluation Log:")
        print(f"   Location: {quality_log}")
        
        # Count queries processed
        with open(quality_log, 'r') as f:
            lines = f.readlines()
            query_count = len([line for line in lines if line.startswith("Query:")])
            print(f"   📝 Processed {query_count} queries")
    
    # List all result files
    print(f"\n📁 All result files in {output_dir}:")
    for file in glob.glob(f"{output_dir}/**/*", recursive=True):
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print(f"   {file} ({size} bytes)")

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser/experiment_results"
    analyze_experiment_logs(output_dir)
EOF
    
    chmod +x $OUTPUT_DIR/analyze_results.py
    python3 $OUTPUT_DIR/analyze_results.py $OUTPUT_DIR
}

# Main execution
main() {
    case "${1:-all}" in
        "baseline")
            cleanup_processes
            check_data
            run_baseline_experiments
            ;;
        "quality")
            check_data
            run_quality_evaluation
            ;;
        "analyze")
            analyze_results
            ;;
        "all")
            cleanup_processes
            check_data
            run_baseline_experiments
            run_quality_evaluation
            analyze_results
            ;;
        *)
            echo "Usage: $0 [baseline|quality|analyze|all]"
            echo "  baseline  - Run performance experiments only"
            echo "  quality   - Run search quality evaluation only"
            echo "  analyze   - Analyze existing results"
            echo "  all       - Run everything (default)"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=== Experiments Complete ==="
    echo "Results available in: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Review logs in $OUTPUT_DIR"
    echo "2. Use your check_url.py to validate search results"
    echo "3. Implement multi-cluster modifications"
}

# Run main function with all arguments
main "$@"