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

func (c *Config) MakeConfigFromPreprocessConfig(preambleStr str, preprocessConfigPath string) error {
	preprocessConfig, err := LoadPreprocessConfig(preprocessConfigPath)
	if err != nil {
		return err
	}

	// Update preamble to point to UUID-specific directory
	c.preamble = filepath.Join(preambleStr, preprocessConfig.UUID)

	return nil
}