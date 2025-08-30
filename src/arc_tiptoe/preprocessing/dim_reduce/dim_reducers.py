"""
Dimensionality reducers collection.
"""

from arc_tiptoe.preprocessing.dim_reduce.dim_reduce import DimReducePCA

dim_reducers = {
    "pca": DimReducePCA,
}
