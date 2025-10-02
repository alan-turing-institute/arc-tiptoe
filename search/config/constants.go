package config

import (
	"math"
)

type Config struct {
	preamble                   string
	imageSearch                bool
	numClusters                int // if > 0, override hardcoded number of clusters
	embeddingDim               int
	searchTopK                 int
	searchConfigPath           string
	numSearchResultsPerCluster int // max results to return per cluster
}

func MakeConfig(preambleStr string, images bool) *Config {
	c := Config{
		preamble:    preambleStr,
		imageSearch: images,
		numClusters: -1, // default to -1, meaning use hardcoded values
	}
	return &c
}

func (c *Config) GetSearchConfigPath() string {
	return c.searchConfigPath
}

func (c *Config) SetSearchConfigPath(path string) {
	c.searchConfigPath = path
}

func (c *Config) GetNumSearchResultsPerCluster() int {
	return c.numSearchResultsPerCluster
}

func (c *Config) SetNumSearchResultsPerCluster(n int) {
	c.numSearchResultsPerCluster = n
}

func (c *Config) GetSearchTopK() int {
	return c.searchTopK
}

func (c *Config) PREAMBLE() string {
	return c.preamble
}

func (c *Config) IMAGE_SEARCH() bool {
	return c.imageSearch
}

// TODO: Fix to be more accurate
func (c *Config) DEFAULT_EMBEDDINGS_HINT_SZ() uint64 {
	if !c.imageSearch {
		return 500
	} else {
		return 900
	}
}

func DEFAULT_URL_HINT_SZ() uint64 {
	return 100
}

func (c *Config) EMBEDDINGS_DIM() uint64 {
	// Use config if available
	if c.embeddingDim > 0 {
		return uint64(c.embeddingDim)
	}

	// Default values
	if !c.imageSearch {
		return 192
	} else {
		return 384
	}
}

func SLOT_BITS() uint64 {
	return 5
}

// Deprecated as hard coded
// func (c *Config) TOTAL_NUM_CLUSTERS() int {
//   if !c.imageSearch {
//     return 1280  // Change from 25196 to 1280
//   } else {
//     return 42528
//   }
// }

func (c *Config) TOTAL_NUM_CLUSTERS() int {
	if c.numClusters > 0 {
		return c.numClusters
	}

	// Otherwise use original hardcoded number for MSMARCO
	if !c.imageSearch {
		return 1280 // Change from 25196 to 1280
	} else {
		return 42528
	}
}

// Round up (# clusters / # embedding servers)
func (c *Config) EMBEDDINGS_CLUSTERS_PER_SERVER() int {
	clustersPerServer := float64(c.TOTAL_NUM_CLUSTERS()) / float64(c.MAX_EMBEDDINGS_SERVERS())
	return int(math.Ceil(clustersPerServer))
}

func (c *Config) MAX_EMBEDDINGS_SERVERS() int {
	totalClusters := c.TOTAL_NUM_CLUSTERS()

	if !c.imageSearch {
		// for text: use 1 server per 16 clusters, up to 80 servers
		servers := (totalClusters + 15) / 16
		if servers < 1 {
			servers = 1
		}
		if servers > 80 {
			servers = 80
		}
		return servers
	} else {
		return 80 // keep images the same for now
	}
}

// Round up (# clusters / # url servers)
func (c *Config) URL_CLUSTERS_PER_SERVER() int {
	clustersPerServer := float64(c.TOTAL_NUM_CLUSTERS()) / float64(c.MAX_URL_SERVERS())
	return int(math.Ceil(clustersPerServer))
}

func (c *Config) MAX_URL_SERVERS() int {
	totalClusters := c.TOTAL_NUM_CLUSTERS()
	if !c.imageSearch {
		// For text: use 1 server per 160 clusters, but cap at 8 like original design
		servers := (totalClusters + 159) / 160
		if servers < 1 {
			servers = 1
		}
		if servers > 8 {
			servers = 8
		}
		return servers
	} else {
		// For images: use 1 server per 528 clusters, but cap at 8 like original design
		return 16 // keep images the same for now
	}
}

func (c *Config) SIMPLEPIR_EMBEDDINGS_RECORD_LENGTH() int {
	if !c.imageSearch {
		return 17
	} else {
		return 15
	}
}
