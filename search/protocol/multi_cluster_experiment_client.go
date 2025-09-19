/*
	Multi cluster experiments client
*/

package protocol

import (
	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/fatih/color"
)

type expQueryClusterResults struct {
	clusterRank  int
	clusterIndex int
	results      []singleQueryClusterResult
	perfUp       float64
	perfDown     float64
}

type expAllClusterResults struct {
	allResults    []expQueryClusterResults
	finalPerfUp   float64
	finalPerfDown float64
}

// Multi cluster experiment will run a series of multi-cluster search queries
// It will return the results, which will then be picked up by the python experiment script
// It run per query, the top K clusters as defined in config
func MultiClusterSearchExperiment(coordinatorAddr string, conf *config.Config, textQuery string, verbose bool) []expAllClusterResults {
	var allExperimentResults expAllClusterResults
	col := color.New(color.FgYellow).Add(color.Bold)
	col.Printf("Running multi-cluster search experiment for query: %s\n", textQuery)

	for topKCluster := 1; topKCluster <= conf.GetSearchTopK(); topKCluster++ {
		color.Cyan("Running multi-cluster search for Cluster %d of %d", topKCluster, conf.GetSearchTopK())
		singleClusterQueryResult := runSingleClusterSearch(coordinatorAddr, conf, verbose, topKCluster, textQuery)
		var expResults expQueryClusterResults
		expResults.clusterRank = topKCluster
		expResults.clusterIndex = singleClusterQueryResult.clusterIndex
		expResults.results = singleClusterQueryResult.results
		expResults.perfUp = singleClusterQueryResult.perf.up1 + singleClusterQueryResult.perf.up2 + singleClusterQueryResult.perf.upOffline
		expResults.perfDown = singleClusterQueryResult.perf.down1 + singleClusterQueryResult.perf.down2 + singleClusterQueryResult.perf.downOffline

		allExperimentResults.allResults = append(allExperimentResults.allResults, expResults)
		if topKCluster == 1 {
			allExperimentResults.finalPerfUp = expResults.perfUp
			allExperimentResults.finalPerfDown = expResults.perfDown
		} else {
			allExperimentResults.finalPerfUp += expResults.perfUp
			allExperimentResults.finalPerfDown += expResults.perfDown
		}
	}

	return []expAllClusterResults{allExperimentResults}
}
