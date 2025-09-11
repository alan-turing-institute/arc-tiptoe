package config

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
)

type PreprocessConfig struct {
    UUID          string `json:"uuid"`
    EmbedModel    string `json:"embed_model"`
    EmbedLib      string `json:"embed_lib"`
    EmbeddingsPath string `json:"embeddings_path"`
    ClusteringPath string `json:"clustering_path"`
    DimRedPath     string `json:"dim_red_path"`
}

func LoadPreprocessConfig(configPath string) (*PreprocessConfig, error) {
	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %v", err)
	}

	var config PreprocessConfig
	err = json.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config JSON: %v", err)
	}

	return &config, nil
}

func (c *Config) MakeConfigFromPreprocessConfig(preambleStr string, preprocessConfigPath string, images bool) error {
	preprocessConfig, err := LoadPreprocessConfig(preprocessConfigPath)
	if err != nil {
		return err
	}

	// Update preamble to point to UUID-specific directory
	c.preamble = filepath.Join(preambleStr, "data", preprocessConfig.UUID)
	c.imageSearch = images

	// Log the paths from the preprocessing config
	fmt.Printf("Using embedding model: %s\n", preprocessConfig.EmbedModel)
	fmt.Printf("Using embedding library: %s\n", preprocessConfig.EmbedLib)
	fmt.Printf("Using embeddings path: %s\n", preprocessConfig.EmbeddingsPath)
	fmt.Printf("Using clustering path: %s\n", preprocessConfig.ClusteringPath)
	fmt.Printf("Using dimensionality reduction path: %s\n", preprocessConfig.DimRedPath)

	return nil
}