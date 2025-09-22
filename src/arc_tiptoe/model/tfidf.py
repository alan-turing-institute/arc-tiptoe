import os
import pickle
import time

from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from arc_tiptoe.constants import MODELS_DIR


def train_tfidf_model(doc_contents):
    """Train TF-IDF model using scikit-learn with time estimation."""
    doc_count = len(doc_contents)

    # Check if model already exists
    existing_model = check_existing_model(doc_count)
    if existing_model:
        print(f"Found existing TF-IDF model for {doc_count} documents")
        return load_tfidf_model(existing_model)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    start_time = time.time()
    tfidf_matrix = vectorizer.fit_transform(doc_contents)
    actual_time = time.time() - start_time

    print(
        f"TF-IDF training completed in {actual_time:.2f}s "
        f"({actual_time / 60:.2f} minutes)"
    )
    print(f"TF-IDF model trained with {tfidf_matrix.shape[1]} features")

    # Save the model
    save_tfidf_model(
        vectorizer, tfidf_matrix, model_name=f"tfidf_model_{doc_count}_docs"
    )

    return vectorizer, tfidf_matrix


def check_existing_model(doc_count):
    """Check if a model exists for the given document count."""
    model_dir = f"{MODELS_DIR}/tfidf_models"
    model_name = f"tfidf_model_{doc_count}_docs"

    vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, f"{model_name}_matrix.npz")

    if os.path.exists(vectorizer_path) and os.path.exists(matrix_path):
        return model_name
    return None


def save_tfidf_model(vectorizer, tfidf_matrix, model_name):
    """Save the trained TF-IDF model to disk."""
    model_dir = f"{MODELS_DIR}/tfidf_models"
    os.makedirs(model_dir, exist_ok=True)

    # Save vectorizer using pickle
    vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # Save TF-IDF matrix using scipy sparse format
    matrix_path = os.path.join(model_dir, f"{model_name}_matrix.npz")
    save_npz(matrix_path, tfidf_matrix)

    print(f"TF-IDF model saved to {model_dir}/")
    print(f"  Vectorizer: {vectorizer_path}")
    print(f"  Matrix: {matrix_path}")


def load_tfidf_model(model_name):
    """Load a saved TF-IDF model from disk."""
    model_dir = f"{MODELS_DIR}/tfidf_models"

    # Load vectorizer
    vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Load TF-IDF matrix
    matrix_path = os.path.join(model_dir, f"{model_name}_matrix.npz")
    tfidf_matrix = load_npz(matrix_path)

    print(f"TF-IDF model loaded from {model_dir}/")
    print(f"  Vectorizer: {vectorizer_path}")
    print(f"  Matrix: {matrix_path}")

    return vectorizer, tfidf_matrix
