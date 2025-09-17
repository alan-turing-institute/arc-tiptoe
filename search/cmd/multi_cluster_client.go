package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/protocol"
)

type SearchResult struct {
	URL       string  `json:"url"`
	Score     float64 `json:"score"`
	ClusterID int     `json:"cluster_id"`
}

type MultiClusterResult struct {
	QueryID     string         `json:"query_id"`
	QueryText   string         `json:"query_text"`
	Results     []SearchResult `json:"results"`
	LatencyMs   float64        `json:"latency_ms"`
	ClusterList []int          `json:"clusters_searched"`
}

var (
	coordinatorAddr = flag.String("coordinator", "127.0.0.1:1237", "Coordinator address")
	configPath      = flag.String("config", "", "Path to config file")
	queryText       = flag.String("query", "", "Query text")
	clusterList     = flag.String("clusters", "", "Comma-separated list of clusters to search")
	topN            = flag.Int("top_n", 10, "Number of top results to return")
	outputFile      = flag.String("output", "", "Output file for results (JSON)")
	verbose         = flag.Bool("verbose", false, "Verbose output")
)

func main() {
	flag.Parse()

	if *configPath == "" || *queryText == "" {
		fmt.Println("Usage: multi_cluster_client -config config.json -query 'search text' [-clusters 1,2,3]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Parse cluster list
	var clusters []int
	if *clusterList != "" {
		clusterStrs := strings.Split(*clusterList, ",")
		for _, clusterStr := range clusterStrs {
			if cluster, err := strconv.Atoi(strings.TrimSpace(clusterStr)); err == nil {
				clusters = append(clusters, cluster)
			}
		}
	}

	// Load config
	conf := &config.Config{}
	if err := conf.LoadFromSearchConfig("../", *configPath); err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		os.Exit(1)
	}

	protocol.RunMultiClusterExperiment(*coordinatorAddr, *queryText, conf.GetSearchTopK(), conf)

	// // Run the search
	// result, err := runMultiClusterSearch(*queryText, clusters, conf)
	// if err != nil {
	// 	fmt.Printf("Search failed: %v\n", err)
	// 	os.Exit(1)
	// }

	// // Output results
	// if *outputFile != "" {
	// 	// Save to file
	// 	file, err := os.Create(*outputFile)
	// 	if err != nil {
	// 		fmt.Printf("Failed to create output file: %v\n", err)
	// 		os.Exit(1)
	// 	}
	// 	defer file.Close()

	// 	encoder := json.NewEncoder(file)
	// 	encoder.SetIndent("", "  ")
	// 	if err := encoder.Encode(result); err != nil {
	// 		fmt.Printf("Failed to write results: %v\n", err)
	// 		os.Exit(1)
	// 	}
	// } else {
	// 	// Print to stdout
	// 	encoder := json.NewEncoder(os.Stdout)
	// 	encoder.SetIndent("", "  ")
	// 	encoder.Encode(result)
	// }

	// if *verbose {
	// 	fmt.Fprintf(os.Stderr, "Search completed: %d results in %.2fms\n",
	// 		len(result.Results), result.LatencyMs)
	// }
}
