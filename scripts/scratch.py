from arc_tiptoe.preprocessing.clustering.clustering import KMeansClusterer
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig

json_config_path = "configs/distilbert_post_embedding.json"
config = PreProcessConfig(json_config_path)

clusterer = KMeansClusterer(config, within_pipeline=True)
print('here')