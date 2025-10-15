/*
	Multi cluster client methods

		Includes:
			- MultiClusterSearchClient: for interactive search client
			- MultiClusterSearchExperiment: for running experiments

	Author: Edmund Dable-Heath
*/

package protocol

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/corpus"
	"github.com/ahenzinger/tiptoe/search/embeddings"
	"github.com/ahenzinger/tiptoe/search/utils"
	"github.com/fatih/color"
)

type singleQueryClusterResult struct {
	Score int    `json:"score"`
	Url   string `json:"url"`
}

type queryClusterResults struct {
	clusterIndex int
	results      []singleQueryClusterResult
	perf         Perf
}

type userQuery struct {
	ClusterIndex       uint64
	Emb                []int8
	TopKClusterIndices []uint64
}

// MultiClusterSearchClient runs an interactive multi-cluster search client
// connecting to the given coordinator address, using the given config
// If verbose is true, prints out detailed step-by-step information

func MultiClusterSearchClient(coordinatorAddr string,
	conf *config.Config,
	verbose bool) {
	for {
		col := color.New(color.FgYellow).Add(color.Bold)
		col.Printf("Enter private search query: ")
		text := utils.ReadLineFromStdin()
		fmt.Printf("\n\n")

		var allQueryResults []queryClusterResults

		// Embed query for multi-cluster search
		color.Cyan("Embedding query and selecting top %d clusters to search over",
			conf.GetSearchTopK())

		// set up embedding process
		in, out := embeddings.SetupEmbeddingProcess(conf.GetNumClusters(), conf)

		// Declare the query data type
		query := new(userQuery)

		// send query to embed process and get back embedding + top clusters to search
		io.WriteString(in, text+"\n")
		if err := json.NewDecoder(out).Decode(&query); err != nil {
			log.Printf("Error in python script processing query: %s", err)
			panic(err)
		}

		color.Green("Query embedded and top clusters selected. Searching over clusters %v\n",
			query.TopKClusterIndices)

		// Check enough clusters are returned
		if len(query.TopKClusterIndices) < conf.GetSearchTopK() {
			panic(fmt.Sprintf("Expected at least %d clusters, got %d",
				conf.GetSearchTopK(),
				len(query.TopKClusterIndices)))
		}

		for topKCluster := 1; topKCluster <= conf.GetSearchTopK(); topKCluster++ {
			color.Cyan("Running multi-cluster search for Cluster %d of %d",
				topKCluster,
				conf.GetSearchTopK())
			searchClusterIndex := query.TopKClusterIndices[topKCluster-1]
			singleClusterQueryResult := runSingleClusterSearch(coordinatorAddr,
				conf,
				verbose,
				query.Emb,
				searchClusterIndex,
				text)
			allQueryResults = append(allQueryResults, singleClusterQueryResult)
		}

		finalResults, finalPerf := parseResultsFromAllClusters(allQueryResults,
			conf.GetNumSearchResultsPerCluster())

		fmt.Printf("\nFinal combined results:\n")
		for i, res := range finalResults {
			fmt.Printf("\t% 3d) [score %s] %s\n", i+1,
				color.YellowString(fmt.Sprintf("% 4d", res.Score)),
				color.BlueString(res.Url))
		}
		fmt.Printf("\n\n")

		fmt.Printf("\nOverall performance metrics across all clusters:\n")
		fmt.Printf("\tAnswered in:\n\t\t%v (preproc)\n\t\t%v (client total)\n\t\t%v (round 1)\n\t\t%v (round 2)\n\t\t%v (total)\n---\n",
			finalPerf.clientPreproc,
			finalPerf.clientSetup,
			finalPerf.t1,
			finalPerf.t2,
			finalPerf.clientTotal)
		fmt.Printf("With total upload %.2f MB and total download %.2f MB\n",
			finalPerf.up1+finalPerf.up2+finalPerf.upOffline,
			finalPerf.down1+finalPerf.down2+finalPerf.downOffline)

	}
}

func parseResultsFromAllClusters(allQueryResults []queryClusterResults,
	numSearchResultsPerCluster int) ([]singleQueryClusterResult, Perf) {
	// Combine results from all clusters
	resultMap := make(map[string]singleQueryClusterResult)
	for _, clusterResults := range allQueryResults {
		for _, res := range clusterResults.results {
			if existing, ok := resultMap[res.Url]; ok {
				if res.Score > existing.Score {
					resultMap[res.Url] = res
				}
			} else {
				resultMap[res.Url] = res
			}
		}
	}

	// Convert map to slice
	var combinedResults []singleQueryClusterResult
	for _, res := range resultMap {
		combinedResults = append(combinedResults, res)
	}

	// Sort by score
	sortedIndices := utils.SortByScores(func() []int {
		scores := make([]int, len(combinedResults))
		for i, res := range combinedResults {
			scores[i] = res.Score
		}
		return scores
	}())

	// Select top results
	var finalResults []singleQueryClusterResult
	for i := 0; i < len(sortedIndices) && i < numSearchResultsPerCluster; i++ {
		finalResults = append(finalResults, combinedResults[sortedIndices[i]])
	}

	// Also compute full performance metrics
	var totalPerf Perf
	for _, clusterResults := range allQueryResults {
		totalPerf.clientPreproc += clusterResults.perf.clientPreproc
		totalPerf.clientSetup += clusterResults.perf.clientSetup
		totalPerf.t1 += clusterResults.perf.t1
		totalPerf.up1 += clusterResults.perf.up1
		totalPerf.down1 += clusterResults.perf.down1
		totalPerf.t2 += clusterResults.perf.t2
		totalPerf.up2 += clusterResults.perf.up2
		totalPerf.down2 += clusterResults.perf.down2
		totalPerf.clientTotal += clusterResults.perf.clientTotal
		totalPerf.upOffline += clusterResults.perf.upOffline
		totalPerf.downOffline += clusterResults.perf.downOffline
		totalPerf.tOffline += clusterResults.perf.tOffline
	}

	return finalResults, totalPerf

}

func runSingleClusterSearch(coordinatorAddr string,
	conf *config.Config,
	verbose bool,
	queryEmb []int8,
	searchClusterIndex uint64,
	text string) queryClusterResults {
	color.Yellow("Setting up client...")

	client := NewClient(true /* use coordinator */)
	client.printStep("Getting metadata")
	hint := client.getHint(true /* keep conn */, coordinatorAddr)
	client.Setup(hint)
	logHintSize(hint)

	client.stepCount = 1
	client.printStep("Running client preprocessing")
	perf := client.preprocessRound(coordinatorAddr,
		true, /* verbose */
		true /* keep conn */)

	queryResults := client.singleClusterSearchRunRound(perf,
		queryEmb,
		coordinatorAddr,
		verbose, true, /* keep conn */
		conf.GetNumSearchResultsPerCluster(),
		searchClusterIndex,
		text)

	return queryResults
}

func (client *Client) singleClusterSearchRunRound(perf Perf,
	queryEmb []int8,
	coordinatorAddr string,
	verbose bool,
	keepConn bool,
	numSearchResultsPerCluster int,
	ClusterIndex uint64,
	text string) queryClusterResults {
	y := color.New(color.FgYellow, color.Bold)
	fmt.Printf("Executing query \"%s\"\n", y.Sprintf(text))

	// Build embeddings query
	start := time.Now()
	if verbose {
		client.printStep("Building embedding query")
	}

	// Check clusters are in range
	if ClusterIndex >= uint64(client.NumClusters()) {
		panic("One of the returned clusters indices is out of range")
	}

	if verbose {
		client.printStep(fmt.Sprintf("Building PIR query for cluster %d",
			ClusterIndex))
	}

	// Embed the query for url server search
	embQuery := client.QueryEmbeddings(queryEmb, ClusterIndex)
	perf.clientSetup = time.Since(start).Seconds()

	// Send embeddings query to server
	if verbose {
		client.printStep("Sending SimplePIR query to server")
	}
	networkingStart := time.Now()
	embAns := client.getEmbeddingsAnswer(embQuery, true /* keep conn */, coordinatorAddr)
	perf.t1, perf.up1, perf.down1 = logStats(client.params.NumDocs,
		networkingStart,
		embQuery,
		embAns)

	// Recover document and URL chunk to query for
	if verbose {
		client.printStep("Decrypting server answer")
	}
	embDec := client.ReconstructEmbeddingsWithinCluster(embAns, ClusterIndex)
	scores := embeddings.SmoothResults(embDec, client.embInfo.P())
	indicesByScore := utils.SortByScores(scores)
	docIndex := indicesByScore[0]
	if verbose {
		fmt.Printf(
			"\tDoc %d within cluster %d has the largest inner produce with out query\n",
			docIndex,
			ClusterIndex)
		client.printStep(
			fmt.Sprintf(
				"Building PIR query for url/title of doc %d in cluster %d",
				docIndex,
				ClusterIndex))
	}

	// Build URL query
	urlQuery, retrievedChunk := client.QueryUrls(ClusterIndex, docIndex)

	// Send URL query to server
	if verbose {
		client.printStep(fmt.Sprintf(
			"Sending PIR query to server for chunk %d",
			retrievedChunk))
	}
	networkingStart = time.Now()
	urlAns := client.getUrlsAnswer(urlQuery, keepConn, coordinatorAddr)
	perf.t2, perf.up2, perf.down2 = logStats(client.params.NumDocs,
		networkingStart,
		urlQuery,
		urlAns)

	// Recover URLs of top n docs in chunk, as defined in config
	urls := client.ReconstructUrls(urlAns, ClusterIndex, docIndex)
	if verbose {
		client.printStep("Reconstructed PIR answers.")
		fmt.Printf("\tTop %d results in cluster %d\n",
			numSearchResultsPerCluster,
			ClusterIndex)
	}

	var queryResults queryClusterResults
	queryResults.clusterIndex = int(ClusterIndex)
	queryResults.results = make([]singleQueryClusterResult, 0)

	j := 1
	for at := 0; at < len(indicesByScore); at++ {
		if scores[at] == 0 {
			break
		}

		doc := indicesByScore[at]
		_, chunk, index := client.urlMap.SubclusterToIndex(ClusterIndex, doc)

		if chunk == retrievedChunk {
			fmt.Printf("\t% 3d) [score %s] %s\n", j,
				color.YellowString(fmt.Sprintf("% 4d", scores[at])),
				color.BlueString(corpus.GetIthUrl(urls, index)))
			queryResults.results = append(queryResults.results,
				singleQueryClusterResult{
					Score: scores[at],
					Url:   corpus.GetIthUrl(urls, index),
				})
		}
		j += 1
		if j > numSearchResultsPerCluster {
			break
		}
	}

	perf.clientTotal = time.Since(start).Seconds()
	fmt.Printf(
		"\tAnswered in:\n\t\t%v (preproc)\n\t\t%v (client total)\n\t\t%v (round 1)\n\t\t%v (round 2)\n\t\t%v (total)\n---\n",
		perf.clientPreproc,
		perf.clientSetup,
		perf.t1,
		perf.t2,
		perf.clientTotal)

	queryResults.perf = perf

	return queryResults
}
