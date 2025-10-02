import logging
from multiprocessing import Pool, cpu_count

import nltk
from ir_datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def get_relevant_docs(
    doc_dict: dict[str, int], target_relevance_level: int | None
) -> list[str]:
    """
    Get relevant document IDs for a given query ID.

    Args:
        doc_dict: A dictionary mapping document IDs to their relevance scores.
        target_relevance_level: The relevance level to filter documents by.

    Returns:
        A list of relevant document IDs.
    """
    if target_relevance_level is not None:
        # If a specific relevance level is targeted, filter for that level
        return [
            doc_id for doc_id, rel in doc_dict.items() if rel == target_relevance_level
        ]
    return [doc_id for doc_id, rel in doc_dict.items() if rel > 0]


def preprocess_text(text: str) -> str:
    """Preprocess text with tokenization, lowercasing, and stemming.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    stemmer = PorterStemmer()
    stop_words = (
        set(stopwords.words("english"))
        if nltk.data.find("corpora/stopwords")
        else set()
    )

    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove stopwords and stem
    processed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            processed_tokens.append(stemmer.stem(token))

    return " ".join(processed_tokens)


def preprocess_batch(texts):
    """Preprocess a batch of texts using multiprocessing."""
    return [preprocess_text(text) for text in texts]


def load_documents_from_ir_datasets(
    dataset: Dataset, max_documents: int | None, batch_size: int = 1000
) -> tuple[list[str], list[str]]:
    """
    Load documents from an ir_datasets dataset.

    Args:
        dataset: The ir_datasets dataset to load documents from.
        max_documents: Maximum number of documents to load. If None, load all.
        batch_size: Number of documents to process in each batch. Defaults to 1000.

    Returns:
        A tuple of two lists: document IDs and document contents.
    """
    doc_ids = []
    doc_contents = []

    batch_texts = []
    batch_ids = []

    with Pool(processes=cpu_count()) as pool:
        for i, doc in tqdm(
            enumerate(dataset.docs_iter()),
            desc="Doc No.",
            total=max_documents,
        ):
            if i >= max_documents:
                break

            batch_ids.append(doc.doc_id)
            # Combine title and body for better retrieval
            full_text = ""
            if hasattr(doc, "title") and doc.title:
                full_text += f"{doc.title}"
            else:
                if hasattr(doc, "text") and doc.text:
                    full_text += doc.text
                elif hasattr(doc, "body") and doc.body:
                    full_text += doc.body
            batch_texts.append(full_text)

            # Process batch when it's full
            if len(batch_texts) >= batch_size:
                # Split batch for parallel processing
                batch_chunks = [
                    batch_texts[j : j + batch_size // cpu_count()]
                    for j in range(0, len(batch_texts), batch_size // cpu_count())
                ]

                # Process chunks in parallel
                processed_chunks = pool.map(preprocess_batch, batch_chunks)

                # Flatten results
                for chunk in processed_chunks:
                    doc_contents.extend(chunk)

                doc_ids.extend(batch_ids)
                batch_texts = []
                batch_ids = []

        # Process remaining documents
        if batch_texts:
            batch_chunks = [
                batch_texts[j : j + batch_size // cpu_count()]
                for j in range(0, len(batch_texts), batch_size // cpu_count())
            ]
            processed_chunks = pool.map(preprocess_batch, batch_chunks)
            for chunk in processed_chunks:
                doc_contents.extend(chunk)
            doc_ids.extend(batch_ids)

    msg = f"Loaded {len(doc_contents)} documents"
    logging.info(msg)

    return doc_ids, doc_contents


def load_doc_ids_only(dataset: Dataset, max_documents=None) -> list[str]:
    """Load only document IDs from ir_datasets, if we don't need full documents.

    Args:
        dataset (Dataset): The ir_datasets dataset object.
        max_documents (int | None): Maximum number of document IDs to load. If None,
        load all.

    Returns:
        List of document IDs."""
    logging.info("Loading only document IDs...")
    doc_ids = []
    n_docs = max_documents if max_documents else dataset.docs_count()
    for i, doc in tqdm(enumerate(dataset.docs_iter()), desc="Doc No.", total=n_docs):
        if max_documents and i >= max_documents:
            break
        doc_ids.append(doc.doc_id)
    msg = f"Loaded {len(doc_ids)} document IDs"
    logging.info(msg)
    return doc_ids


def load_queries_from_ir_datasets(
    dataset: Dataset,
) -> list[tuple[str, str, str, dict[str, int]]]:
    """
    Load queries from an ir_datasets dataset.

    Args:
        dataset: The ir_datasets dataset to load queries from.

    Returns:
        A list of tuples: (query_id, original_query, processed_query, qrels_dict).
    """
    qrels_ref = dataset.qrels_dict()
    query_list = []
    for query in dataset.queries_iter():
        query_id = query.query_id
        qrels = qrels_ref.get(query_id)
        if qrels is None:
            continue  # Skip queries without qrels
        query_text = query.text
        processed_query = preprocess_text(query_text)
        query_list.append((query_id, query_text, processed_query, qrels))

    msg = f"Loaded {len(query_list)} queries"
    logging.info(msg)
    return query_list
