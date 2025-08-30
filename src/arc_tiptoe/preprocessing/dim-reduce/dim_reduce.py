"""
Class for dimensionality reduction methods.
"""


class DimReduce:
    """Base class for all dimensionality reduction methods."""

    def __init__(self, config):
        self.config = config

    def reduce_dimensions(self, embeddings):
        """Reduce the dimensions of the embeddings."""
        raise NotImplementedError("This method should be overridden by subclasses.")
