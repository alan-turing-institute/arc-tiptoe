import logging
from multiprocessing import Pool, cpu_count

import ir_datasets
import nltk
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
        qrels_ref: A reference to the qrels dictionary.
        query_id: The ID of the query.

    Returns:
        A list of relevant document IDs.
    """
    if target_relevance_level is not None:
        # If a specific relevance level is targeted, filter for that level
        return [
            doc_id for doc_id, rel in doc_dict.items() if rel == target_relevance_level
        ]
    return [doc_id for doc_id, rel in doc_dict.items() if rel > 0]


def preprocess_text(text):
    """Preprocess text with tokenization, lowercasing, and stemming."""
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


def load_documents_from_ir_datasets(dataset_name, max_documents, batch_size=1000):
    dataset = ir_datasets.load(dataset_name)

    doc_ids = []
    doc_contents = []

    msg = f"Processing {dataset_name} documents in batches..."
    logging.info(msg)

    batch_texts = []
    batch_ids = []

    n_docs = dataset.docs_count() if max_documents == "full" else max_documents

    with Pool(processes=cpu_count()) as pool:
        for i, doc in tqdm(
            enumerate(dataset.docs_iter()), desc="Loading", total=n_docs
        ):
            if max_documents != "full" and i >= max_documents:
                break

            batch_ids.append(doc.doc_id)
            # Combine title and body for better retrieval
            if hasattr(doc, "title") and doc.title:
                full_text = f"{doc.title} {doc.body}"
            else:
                full_text = doc.body
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


def load_doc_ids_only(max_documents=None, dataset_name="msmarco-document/trec-dl-2019"):
    """Load only document IDs from ir_datasets."""
    logging.info("Loading only document IDs...")
    dataset = ir_datasets.load(dataset_name)
    doc_ids = []
    n_docs = max_documents if max_documents else dataset.docs_count()
    for i, doc in tqdm(enumerate(dataset.docs_iter()), desc="Doc IDs", total=n_docs):
        if max_documents and i >= max_documents:
            break
        doc_ids.append(doc.doc_id)
    msg = f"Loaded {len(doc_ids)} document IDs"
    logging.info(msg)
    return doc_ids


def load_queries_from_ir_datasets(dataset_name="msmarco-document/trec-dl-2019"):
    dataset = ir_datasets.load(dataset_name)
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
