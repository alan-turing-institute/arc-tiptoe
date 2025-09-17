package protocol

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/embeddings"
)

type MultiClusterSearchResult struct {
	URL       string  `json:"url"`
	Score     float64 `json:"score"`
	ClusterID int     `json:"cluster_id"`
}

type MultiClusterResult struct {
	QueryID     string                     `json:"query_id"`
	QueryText   string                     `json:"query_text"`
	Results     []MultiClusterSearchResult `json:"results"`
	LatencyMs   float64                    `json:"latency_ms"`
	ClusterList []int                      `json:"clusters_searched"`
}

func RunMultiClusterExperiment(coordinatorAddr, queryText string, maxClusters int, conf *config.Config) {
	fmt.Printf("🔍 Multi-cluster experiment: '%s' (max %d clusters)\n", queryText, maxClusters)

	// Setup client once
	client := NewClient(true)
	hint := client.getHint(true, coordinatorAddr)
	client.Setup(hint)

	// Setup embedding process
	in, out := embeddings.SetupEmbeddingProcess(client.NumClusters(), conf)
	defer in.Close()
	defer out.Close()

	// Run search for different cluster counts
	for numClusters := 1; numClusters <= maxClusters; numClusters++ {
		fmt.Printf("   🎯 Testing %d clusters\n", numClusters)

		startTime := time.Now()

		// Use the preprocessing and search
		perf := client.preprocessRound(coordinatorAddr, false, true)
		queryResult, err := client.runRound(perf, in, out, queryText, coordinatorAddr, false, true)

		if err != nil {
			fmt.Printf("    Error: %v\n", err)
			continue
		}

		latencyMs := float64(time.Since(startTime).Nanoseconds()) / 1e6
		throughputQPS := 1000.0 / latencyMs

		fmt.Printf("    %d clusters: %.1fms, %.2f QPS, %d results\n",
			numClusters, latencyMs, throughputQPS, len(queryResult.Results))

		// Output structured results that Python can parse
		fmt.Printf("CLUSTER_COUNT:%d\n", numClusters)
		fmt.Printf("LATENCY_MS:%.2f\n", latencyMs)
		fmt.Printf("THROUGHPUT_QPS:%.2f\n", throughputQPS)
		fmt.Printf("RESULTS_START\n")

		// Output each result in a parseable format
		for i, result := range queryResult.Results {
			if i >= 10 { // Limit to top 10 results
				break
			}
			fmt.Printf("RESULT:%s:%.2f:%d\n", result.URL, result.Score, result.ClusterID)
		}

		fmt.Printf("RESULTS_END\n")
		fmt.Printf("---\n")
	}
}

// func runActualSearch(client *Client, queryText string, targetCluster int, in io.WriteCloser, out io.ReadCloser, coordinatorAddr string) []MultiClusterSearchResult {
// 	// This runs the actual Tiptoe search protocol like the original client

// 	// First do preprocessing (like the original client)
// 	perf := client.preprocessRound(coordinatorAddr, false /* verbose */, true /* keep conn */)

// 	// Run the actual search round (simplified version of runRound)
// 	start := time.Now()

// 	// Get embedding for the query
// 	var query struct {
// 		Cluster_index uint64
// 		Emb           []int8
// 	}

// 	// Send query to embedding process
// 	fmt.Fprintf(in, "%s\n", queryText)

// 	// Read embedding response
// 	decoder := json.NewDecoder(out)
// 	if err := decoder.Decode(&query); err != nil {
// 		fmt.Printf("Error decoding embedding: %v\n", err)
// 		return []MultiClusterSearchResult{}
// 	}

// 	// Build and send embeddings query
// 	embQuery := client.QueryEmbeddings(query.Emb, query.Cluster_index)
// 	embAns := client.getEmbeddingsAnswer(embQuery, true /* keep conn */, coordinatorAddr)

// 	// Recover document within cluster
// 	embDec := client.ReconstructEmbeddingsWithinCluster(embAns, query.Cluster_index)
// 	scores := embeddings.SmoothResults(embDec, client.embInfo.P())
// 	indicesByScore := utils.SortByScores(scores)

// 	results := []MultiClusterSearchResult{}

// 	// Get top 10 results
// 	for i := 0; i < 10 && i < len(indicesByScore) && scores[indicesByScore[i]] > 0; i++ {
// 		docIndex := indicesByScore[i]

// 		// Build URL query for this document
// 		urlQuery, retrievedChunk := client.QueryUrls(query.Cluster_index, docIndex)
// 		urlAns := client.getUrlsAnswer(urlQuery, true /* keep conn */, coordinatorAddr)
// 		urls := client.ReconstructUrls(urlAns, query.Cluster_index, docIndex)

// 		// Extract the URL for this specific document
// 		_, chunk, index := client.urlMap.SubclusterToIndex(query.Cluster_index, docIndex)
// 		if chunk == retrievedChunk {
// 			url := corpus.GetIthUrl(urls, index)

// 			results = append(results, MultiClusterSearchResult{
// 				URL:       url,
// 				Score:     float64(scores[docIndex]),
// 				ClusterID: int(query.Cluster_index),
// 			})
// 		}
// 	}

// 	return results
// }

func getTopClustersFromEmbedding(queryText string, topK int, in io.WriteCloser, out io.ReadCloser) ([]int, error) {
	// Send query to embedding process to get top-k clusters
	if _, err := fmt.Fprintf(in, "%s\n", queryText); err != nil {
		return nil, err
	}

	// Read response
	var response struct {
		Cluster_index  uint64 `json:"Cluster_index"`
		Top_k_clusters []int  `json:"Top_k_clusters"`
	}

	decoder := json.NewDecoder(out)
	if err := decoder.Decode(&response); err != nil {
		return nil, err
	}

	if len(response.Top_k_clusters) > 0 {
		return response.Top_k_clusters[:topK], nil
	}
	return []int{int(response.Cluster_index)}, nil
}
