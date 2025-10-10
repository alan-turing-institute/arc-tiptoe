package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type SearchConfig struct {
	UUID     string `json:"uuid"`
	DataPath string `json:"data_path"`

	Embedding struct {
		ModelName    string `json:"model_name"`
		EmbeddingDim int    `json:"embedding_dim"`
		ReducedDim   int    `json:"reduced_dimension"`
	} `json:"embedding"`

	Clustering struct {
		TotalClusters    int    `json:"total_clusters"`
		SearchTopK       int    `json:"search_top_k"`
		CentroidsFile    string `json:"centroids_file"`
		ClusterDirectory string `json:"cluster_directory"`
	}

	DimRed struct {
		Applied bool   `json:"applied"`
		Method  string `json:"method"`
	} `json:"dimensionality_reduction"`

	ServerConfig struct {
		EmbeddingServers           int `json:"embedding_servers"`
		URLServers                 int `json:"url_servers"`
		ClustersPerEmbeddingServer int `json:"clusters_per_embedding_server"`
		ClustersPerURLServer       int `json:"clusters_per_url_server"`
		EmbeddingHintSize          int `json:"embedding_hint_size"`
		URLHintSize                int `json:"url_hint_size"`
	} `json:"server_config"`

	Artifacts struct {
		FAISSIndex        string `json:"faiss_index"`
		ArtifactDirectory string `json:"artifact_directory"`
	} `json:"artifacts"`

	Search struct {
		MaxResults int `json:"max_results"`
	} `json:"search"`
}

func LoadSearchConfig(configPath string) (*SearchConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read search config: %v", err)
	}

	var config SearchConfig
	err = json.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse search config: %v", err)
	}

	return &config, nil
}

func (c *Config) LoadFromSearchConfig(preambleStr string, searchConfigPath string) error {
	// Store config path
	c.searchConfigPath = searchConfigPath

	searchConfig, err := LoadSearchConfig(searchConfigPath)
	if err != nil {
		return err
	}

	// Set all config values from search config
	c.preamble = filepath.Join(preambleStr, searchConfig.DataPath)
	c.numClusters = searchConfig.Clustering.TotalClusters
	c.embeddingDim = searchConfig.Embedding.ReducedDim
	c.searchTopK = searchConfig.Clustering.SearchTopK
	c.numSearchResultsPerCluster = searchConfig.Search.MaxResults
	c.numEmbedServers = searchConfig.ServerConfig.EmbeddingServers

	// Log configuration
	fmt.Printf("Search Configuration:\n")
	fmt.Printf("  UUID: %s\n", searchConfig.UUID)
	fmt.Printf("  Model: %s\n", searchConfig.Embedding.ModelName)
	fmt.Printf("  Total clusters: %d\n", searchConfig.Clustering.TotalClusters)
	fmt.Printf("  Search top-k clusters: %d\n", searchConfig.Clustering.SearchTopK)
	fmt.Printf("  Embedding dimension: %d -> %d\n", searchConfig.Embedding.EmbeddingDim, searchConfig.Embedding.ReducedDim)
	fmt.Printf("  Embedding servers: %d\n", searchConfig.ServerConfig.EmbeddingServers)
	fmt.Printf("  URL servers: %d\n", searchConfig.ServerConfig.URLServers)

	return nil
}
