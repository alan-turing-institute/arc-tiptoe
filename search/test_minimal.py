#!/usr/bin/env python3
import sys
import os
import json

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test basic imports
print("Testing basic imports...")
try:
    import numpy as np
    print("✅ NumPy OK")
    
    import faiss
    print("✅ FAISS OK")
    
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers OK")
    
    # Test config loading
    config_path = sys.argv[1] if len(sys.argv) > 1 else "../configs/GemmaEmbed_test_config_checkpoint_20250916-114911_search_config.json"
    print(f"Testing config loading: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("✅ Config loaded OK")
    print(f"Model: {config.get('embedding', {}).get('model_name', 'NOT_FOUND')}")
    
    # Test model loading with a simpler model first
    print("Testing model loading...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
    print("✅ Model loaded OK")
    
    # Test encoding
    result = model.encode(["test query"])
    print(f"✅ Encoding OK: shape {result.shape}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
