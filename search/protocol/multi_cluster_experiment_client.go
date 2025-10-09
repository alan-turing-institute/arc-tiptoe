/*
	Multi cluster experiments client
*/

package protocol

import (
	"encoding/json"
	"fmt"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/fatih/color"
)

type expQueryClusterResults struct {
	ClusterRank  int                        `json:"cluster_rank"`
	ClusterIndex int                        `json:"cluster_index"`
	Results      []singleQueryClusterResult `json:"results"`
	PerfUp       float64                    `json:"perf_up"`
	PerfDown     float64                    `json:"perf_down"`
}

type expAllClusterResults struct {
	AllResults    []expQueryClusterResults `json:"all_results"`
	FinalPerfUp   float64                  `json:"final_perf_up"`
	FinalPerfDown float64                  `json:"final_perf_down"`
}

// Multi cluster experiment will run a series of multi-cluster search queries
// It will return the results, which will then be picked up by the
// python experiment script
// It run per query, the top K clusters as defined in config
// This method assumes the query has already been embedded
// and the top K cluster indices selected
// If verbose is true, prints out detailed step-by-step information
// Returns the results of the experiment
func MultiClusterSearchExperiment(coordinatorAddr string,
	conf *config.Config,
	queryEmb []int8,
	topKClusterIndices []uint64,
	queryText string,
	verbose bool) expAllClusterResults {
	var allExperimentResults expAllClusterResults
	if verbose {
		col := color.New(color.FgYellow).Add(color.Bold)
		col.Printf("Running multi-cluster search experiment for query: %s\n", queryText)
	}

	for topKCluster := 1; topKCluster <= conf.GetSearchTopK(); topKCluster++ {
		if verbose {
			color.Cyan("Running multi-cluster search for Cluster %d of %d",
				topKCluster,
				conf.GetSearchTopK())
		}
		searchClusterIndex := topKClusterIndices[topKCluster-1]
		singleClusterQueryResult := runSingleClusterSearch(coordinatorAddr,
			conf,
			verbose,
			queryEmb,
			searchClusterIndex,
			queryText)
		var expResults expQueryClusterResults
		expResults.ClusterRank = topKCluster
		expResults.ClusterIndex = singleClusterQueryResult.clusterIndex
		expResults.Results = singleClusterQueryResult.results
		expResults.PerfUp = singleClusterQueryResult.perf.up1 +
			singleClusterQueryResult.perf.up2 +
			singleClusterQueryResult.perf.upOffline
		expResults.PerfDown = singleClusterQueryResult.perf.down1 +
			singleClusterQueryResult.perf.down2 +
			singleClusterQueryResult.perf.downOffline

		allExperimentResults.AllResults = append(allExperimentResults.AllResults,
			expResults)
		if topKCluster == 1 {
			allExperimentResults.FinalPerfUp = expResults.PerfUp
			allExperimentResults.FinalPerfDown = expResults.PerfDown
		} else {
			allExperimentResults.FinalPerfUp += expResults.PerfUp
			allExperimentResults.FinalPerfDown += expResults.PerfDown
		}
	}

	jExpRes, _ := json.Marshal(allExperimentResults)
	fmt.Println(string(jExpRes))

	return allExperimentResults
}
