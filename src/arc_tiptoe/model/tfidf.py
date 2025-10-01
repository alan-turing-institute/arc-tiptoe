import logging
import os
import pickle
import time
from typing import NamedTuple

from scipy.sparse import load_npz, save_npz, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from arc_tiptoe.constants import MODELS_DIR

EQUIV_DATASETS = {
    "msmarco-document_trec-dl-2019": "msmarco-document_dev",
    "msmarco-document_dev": "msmarco-document_trec-dl-2019",
}


class TFIDFModel(NamedTuple):
    vectorizer: TfidfVectorizer  # Set default value
    tfidf_matrix: spmatrix


def train_tfidf_model(dataset_name, doc_contents):
    """Train TF-IDF model using scikit-learn with time estimation."""
    doc_count = len(doc_contents)

    # Check if model already exists
    existing_model = check_existing_model(
        dataset_name=dataset_name, doc_count=doc_count
    )
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
        vectorizer,
        tfidf_matrix,
        model_name=f"tfidf_model_{doc_count}_docs",
        dataset_name=dataset_name,
    )

    return TFIDFModel(vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)


def check_existing_model(dataset_name, doc_count, accept_equiv=True):
    """Wrapper to check for existing model, considering equivalent datasets."""
    existing_model = _check_existing_model(dataset_name, doc_count)
    if existing_model:
        return existing_model

    # Check equivalent dataset if applicable
    equiv_dataset = EQUIV_DATASETS.get(dataset_name)
    if equiv_dataset and accept_equiv:
        log_msg = (
            f"No model found for {dataset_name}, with {doc_count} documents. "
            f"Found dataset with same documents, checking for model in {equiv_dataset}"
        )
        logging.info(log_msg)
        return _check_existing_model(equiv_dataset, doc_count)

    return None


def _check_existing_model(dataset_name, doc_count):
    """Check if a model exists for the given document count."""
    data_save_name = dataset_name.replace("/", "_")
    model_dir = f"{MODELS_DIR}/{data_save_name}/tfidf"
    model_name = f"tfidf_model_{doc_count}_docs"

    vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.pkl")
    matrix_path = os.path.join(model_dir, f"{model_name}_matrix.npz")

    if os.path.exists(vectorizer_path) and os.path.exists(matrix_path):
        return os.path.join(model_dir, model_name)
    return None


def save_tfidf_model(vectorizer, tfidf_matrix, model_name, dataset_name):
    """Save the trained TF-IDF model to disk."""
    data_save_name = dataset_name.replace("/", "_")
    model_dir = f"{MODELS_DIR}/{data_save_name}/tfidf"
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


def load_tfidf_model(model_path_stem):
    """Load a saved TF-IDF model from disk."""

    # Load vectorizer
    vectorizer_path = f"{model_path_stem}_vectorizer.pkl"
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Load TF-IDF matrix
    matrix_path = f"{model_path_stem}_matrix.npz"
    tfidf_matrix = load_npz(matrix_path)

    msg = f"""TF-IDF model loaded from:
    {os.path.dirname(model_path_stem)}
        Vectorizer: {vectorizer_path}
        Matrix: {matrix_path}
    """

    logging.info(msg)

    return TFIDFModel(vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)
