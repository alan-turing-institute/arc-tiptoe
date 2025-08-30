"""
Embedders collection.
"""

from arc_tiptoe.preprocessing.embedding.embedding import SentenceTransformerEmbedder

embedders = {
    "sentence_transformer": SentenceTransformerEmbedder,
}
