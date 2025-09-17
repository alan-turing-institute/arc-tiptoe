package protocol

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"time"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/embeddings"
	"github.com/ahenzinger/tiptoe/search/utils"
)

type MultiClusterSearchResult struct {
	URL       string  `json:"url"`
	Score     float64 `json:"score"`
	ClusterID int     `json:"cluster_id"`
	Rank      int     `json:"rank"`
}

type MultiClusterResult struct {
	QueryID     string                     `json:"query_id"`
	QueryText   string                     `json:"query_text"`
	Results     []MultiClusterSearchResult `json:"results"`
	LatencyMs   float64                    `json:"latency_ms"`
	CommMB      float64                    `json:"communication_mb"`
	ClusterList []int                      `json:"clusters_searched"`
}

func RunMultiClusterExperiment(coordinatorAddr, queryText string, maxClusters int, conf *config.Config) {
	fmt.Printf("Multi-cluster experiment: '%s' (max %d clusters)\n", queryText, maxClusters)

	// Setup client once
	client := NewClient(true)
	hint := client.getHint(true, coordinatorAddr)
	client.Setup(hint)

	// Setup embedding process with top-k support
	in, out := embeddings.SetupEmbeddingProcessWithTopK(client.NumClusters(), maxClusters, conf)
	defer in.Close()
	defer out.Close()

	// Get embedding and top-k clusters for the query
	topClusters, embedding, err := getEmbeddingAndTopClusters(queryText, maxClusters, in, out)
	if err != nil {
		fmt.Printf("Error getting embedding: %v\n", err)
		return
	}

	fmt.Printf("   Top %d clusters: %v\n", len(topClusters), topClusters)

	// Run search for 1 to k clusters, measuring each
	for numClusters := 1; numClusters <= len(topClusters) && numClusters <= maxClusters; numClusters++ {
		clustersToSearch := topClusters[:numClusters]

		fmt.Printf("   🎯 Testing %d clusters: %v\n", numClusters, clustersToSearch)

		startTime := time.Now()

		// Search each cluster and collect results + communication costs
		allResults := []MultiClusterSearchResult{}
		totalCommMB := 0.0

		for _, clusterID := range clustersToSearch {
			fmt.Printf("     Searching cluster %d...\n", clusterID)
			results, commMB, err := searchSingleClusterSimple(client, embedding, uint64(clusterID), coordinatorAddr)
			if err != nil {
				fmt.Printf("   Warning: cluster %d failed: %v\n", clusterID, err)
				continue
			}

			fmt.Printf("     Cluster %d returned %d results\n", clusterID, len(results))
			allResults = append(allResults, results...)
			totalCommMB += commMB
		}

		// Sort all results by score across all clusters
		sort.Slice(allResults, func(i, j int) bool {
			return allResults[i].Score > allResults[j].Score
		})

		// Update global ranks and limit to top 10
		for i := range allResults {
			allResults[i].Rank = i + 1
		}
		if len(allResults) > 10 {
			allResults = allResults[:10]
		}

		latencyMs := float64(time.Since(startTime).Nanoseconds()) / 1e6

		fmt.Printf("   %d clusters: %.1fms, %.6f MB comm, %d total results\n",
			numClusters, latencyMs, totalCommMB, len(allResults))

		// Output structured results for Python to parse
		fmt.Printf("CLUSTER_COUNT:%d\n", numClusters)
		fmt.Printf("LATENCY_MS:%.2f\n", latencyMs)
		fmt.Printf("COMMUNICATION_MB:%.6f\n", totalCommMB)
		fmt.Printf("RESULTS_START\n")

		for _, result := range allResults {
			fmt.Printf("RESULT:%s:%.2f:%d\n", result.URL, result.Score, result.ClusterID)
		}

		fmt.Printf("RESULTS_END\n")
		fmt.Printf("---\n")
	}
}

func getEmbeddingAndTopClusters(queryText string, topK int, in io.WriteCloser, out io.ReadCloser) ([]int, []int8, error) {
	// Send query to embedding process
	if _, err := fmt.Fprintf(in, "%s\n", queryText); err != nil {
		return nil, nil, err
	}

	// Read response
	var response struct {
		Cluster_index  uint64 `json:"Cluster_index"`
		Emb            []int8 `json:"Emb"`
		Top_k_clusters []int  `json:"Top_k_clusters"`
	}

	decoder := json.NewDecoder(out)
	if err := decoder.Decode(&response); err != nil {
		return nil, nil, err
	}

	clusters := response.Top_k_clusters
	if len(clusters) == 0 {
		clusters = []int{int(response.Cluster_index)}
	}

	return clusters, response.Emb, nil
}

func searchSingleClusterSimple(client *Client, embedding []int8, clusterID uint64, coordinatorAddr string) ([]MultiClusterSearchResult, float64, error) {
	totalCommMB := 0.0

	// 1. Query embeddings for this cluster
	embQuery := client.QueryEmbeddings(embedding, clusterID)

	networkingStart := time.Now()
	embAns := client.getEmbeddingsAnswer(embQuery, true, coordinatorAddr)

	// Track embedding communication
	_, uploadMB, downloadMB := logStats(client.NumDocs(), networkingStart, embQuery, embAns)
	totalCommMB += uploadMB + downloadMB

	// 2. Reconstruct embeddings within cluster
	embDec := client.ReconstructEmbeddingsWithinCluster(embAns, clusterID)
	scores := embeddings.SmoothResults(embDec, client.embInfo.P())
	indicesByScore := utils.SortByScores(scores)

	results := []MultiClusterSearchResult{}

	// 3. Get URLs for top documents with robust error handling
	maxResults := 10
	for i := 0; i < len(indicesByScore) && i < maxResults && scores[indicesByScore[i]] > 0; i++ {
		docIndex := indicesByScore[i]
		score := scores[docIndex]

		// Try to get the actual URL with comprehensive error handling
		url, urlCommMB := client.safeReconstructUrl(clusterID, docIndex, coordinatorAddr)
		totalCommMB += urlCommMB

		// Always add a result, even if URL reconstruction failed
		results = append(results, MultiClusterSearchResult{
			URL:       url,
			Score:     float64(score),
			ClusterID: int(clusterID),
			Rank:      i + 1,
		})
	}

	return results, totalCommMB, nil
}
