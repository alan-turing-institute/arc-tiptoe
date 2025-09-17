package protocol

import (
	"encoding/json"
	"fmt"
	"io"
	"net/rpc"
	"sort"
	"strings"
	"time"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/corpus"
	"github.com/ahenzinger/tiptoe/search/database"
	"github.com/ahenzinger/tiptoe/search/embeddings"
	"github.com/ahenzinger/tiptoe/search/utils"
	"github.com/ahenzinger/underhood/underhood"
	"github.com/fatih/color"
	"github.com/henrycg/simplepir/matrix"
	"github.com/henrycg/simplepir/pir"
)

type UnderhoodAnswer struct {
	EmbAnswer underhood.HintAnswer
	UrlAnswer underhood.HintAnswer
}

type QueryType interface {
	bool | underhood.HintQuery | pir.Query[matrix.Elem64] | pir.Query[matrix.Elem32]
}

type AnsType interface {
	TiptoeHint | UnderhoodAnswer | pir.Answer[matrix.Elem64] | pir.Answer[matrix.Elem32]
}

type SearchResult struct {
	URL       string  `json:"url"`
	Score     float64 `json:"score"`
	ClusterID uint64  `json:"cluster_id"`
	Rank      int     `json:"rank"`
}

type QueryResult struct {
	QueryText    string         `json:"query_text"`
	TopClusters  []int          `json:"top_clusters"`
	Results      []SearchResult `json:"results"`
	LatencyMs    float64        `json:"latency_ms"`
	TotalResults int            `json:"total_results"`
}

type Client struct {
	params corpus.Params

	embClient  *underhood.Client[matrix.Elem64]
	embInfo    *pir.DBInfo
	embMap     database.ClusterMap
	embIndices map[uint64]bool

	urlClient  *underhood.Client[matrix.Elem32]
	urlInfo    *pir.DBInfo
	urlMap     database.SubclusterMap
	urlIndices map[uint64]bool

	rpcClient      *rpc.Client
	useCoordinator bool

	stepCount int
}

func NewClient(useCoordinator bool) *Client {
	c := new(Client)
	c.useCoordinator = useCoordinator
	return c
}

func (c *Client) Free() {
	c.urlClient.Free()
	c.embClient.Free()
}

func (c *Client) NumDocs() uint64 {
	return c.params.NumDocs
}

func (c *Client) NumClusters() int {
	if len(c.embMap) > 0 {
		return len(c.embMap)
	}
	return len(c.urlMap)
}

func (c *Client) printStep(text string) {
	col := color.New(color.FgGreen).Add(color.Bold)
	col.Printf("%d) %v\n", c.stepCount, text)
	c.stepCount += 1
}

func RunClient(coordinatorAddr string, conf *config.Config) {
	color.Yellow("Setting up client...")

	c := NewClient(true /* use coordinator */)
	c.printStep("Getting metadata")
	hint := c.getHint(true /* keep conn */, coordinatorAddr)
	c.Setup(hint)
	logHintSize(hint)

	in, out := embeddings.SetupEmbeddingProcess(c.NumClusters(), conf)
	col := color.New(color.FgYellow).Add(color.Bold)

	for {
		c.stepCount = 1
		c.printStep("Running client preprocessing")
		perf := c.preprocessRound(coordinatorAddr, true /* verbose */, true /* keep conn */)

		col.Printf("Enter private search query: ")
		text := utils.ReadLineFromStdin()
		fmt.Printf("\n\n")
		if (strings.TrimSpace(text) == "") || (strings.TrimSpace(text) == "quit") {
			break
		}
		_, err := c.runRound(perf, in, out, text, coordinatorAddr, true /* verbose */, true /* keep conn */)
		if err != nil {
			fmt.Printf("Query failed: %v\n", err)
		}
	}

	if c.rpcClient != nil {
		c.rpcClient.Close()
	}
	in.Close()
	out.Close()
}

func (c *Client) preprocessRound(coordinatorAddr string, verbose, keepConn bool) Perf {
	var p Perf

	// Perform preprocessing
	start := time.Now()
	ct := c.PreprocessQuery()

	networkingStart := time.Now()
	offlineAns := c.applyHint(ct, keepConn, coordinatorAddr)
	p.tOffline, p.upOffline, p.downOffline = logOfflineStats(c.NumDocs(), networkingStart, ct, offlineAns)

	c.ProcessHintApply(offlineAns)

	p.clientPreproc = time.Since(start).Seconds()

	if verbose {
		fmt.Printf("\tPreprocessing complete -- %fs\n\n", p.clientPreproc)
	}

	return p
}

func (c *Client) runRound(p Perf, in io.WriteCloser, out io.ReadCloser,
	text, coordinatorAddr string, verbose, keepConn bool) (*QueryResult, error) {

	startTime := time.Now()

	if verbose {
		c.printStep("Generating embedding of the query")
	}

	// Get embedding and cluster info from Python process
	var query struct {
		Cluster_index  uint64 `json:"Cluster_index"`
		Emb            []int8 `json:"Emb"`
		Top_k_clusters []int  `json:"Top_k_clusters"`
	}

	fmt.Printf("DEBUG: about to run python subprotocol, '%s'\n", text)

	// Send query to embedding process
	_, err := io.WriteString(in, text+"\n")
	if err != nil {
		return nil, fmt.Errorf("error writing to python process: %v", err)
	}

	fmt.Printf("DEBUG: Query sent, waiting for response...\n")

	decoder := json.NewDecoder(out)
	if err := decoder.Decode(&query); err != nil {
		return nil, fmt.Errorf("error decoding JSON from Python: %v", err)
	}

	fmt.Printf("DEBUG: Successfully received response: cluster=%d, emb_len=%d, top_clusters=%v\n",
		query.Cluster_index, len(query.Emb), query.Top_k_clusters)

	// Determine which clusters to search
	clustersToSearch := []uint64{query.Cluster_index} // Default to primary cluster
	if len(query.Top_k_clusters) > 0 {
		clustersToSearch = make([]uint64, len(query.Top_k_clusters))
		for i, cluster := range query.Top_k_clusters {
			clustersToSearch[i] = uint64(cluster)
		}
	}

	// Collect all results from all clusters
	allResults := []SearchResult{}

	for _, clusterID := range clustersToSearch {
		if verbose {
			fmt.Printf("Searching cluster %d\n", clusterID)
		}

		// Run search protocol for this cluster
		clusterResults, err := c.searchCluster(query.Emb, clusterID, coordinatorAddr, keepConn, verbose)
		if err != nil {
			if verbose {
				fmt.Printf("Warning: failed to search cluster %d: %v\n", clusterID, err)
			}
			continue
		}

		allResults = append(allResults, clusterResults...)
	}

	// Sort all results by score
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})

	// Update ranks after sorting
	for i := range allResults {
		allResults[i].Rank = i + 1
	}

	latencyMs := float64(time.Since(startTime).Nanoseconds()) / 1e6

	result := &QueryResult{
		QueryText:    text,
		TopClusters:  query.Top_k_clusters,
		Results:      allResults,
		LatencyMs:    latencyMs,
		TotalResults: len(allResults),
	}

	if verbose {
		fmt.Printf("Query completed: %d results in %.2fms\n", len(allResults), latencyMs)
	}

	return result, nil
}

// New method to search a specific cluster
func (c *Client) searchCluster(embedding []int8, clusterID uint64, coordinatorAddr string, keepConn, verbose bool) ([]SearchResult, error) {
	if verbose {
		c.printStep(fmt.Sprintf("Querying embeddings for cluster %d", clusterID))
	}

	// Build embeddings query
	embQuery := c.QueryEmbeddings(embedding, clusterID)
	embAns := c.getEmbeddingsAnswer(embQuery, keepConn, coordinatorAddr)

	if verbose {
		c.printStep(fmt.Sprintf("Reconstructing embeddings for cluster %d", clusterID))
	}

	// Recover documents within cluster
	embDec := c.ReconstructEmbeddingsWithinCluster(embAns, clusterID)
	scores := embeddings.SmoothResults(embDec, c.embInfo.P())
	indicesByScore := utils.SortByScores(scores)

	results := []SearchResult{}

	// Get top documents (limit to reasonable number)
	maxResults := 20
	for i := 0; i < len(indicesByScore) && i < maxResults && scores[indicesByScore[i]] > 0; i++ {
		docIndex := indicesByScore[i]
		score := scores[docIndex]

		if verbose {
			c.printStep(fmt.Sprintf("Querying URLs for doc %d in cluster %d", docIndex, clusterID))
		}

		// Get URL for this document
		urlQuery, retrievedChunk := c.QueryUrls(clusterID, docIndex)
		urlAns := c.getUrlsAnswer(urlQuery, keepConn, coordinatorAddr)
		urls := c.ReconstructUrls(urlAns, clusterID, docIndex)

		// Extract the specific URL for this document
		_, chunk, index := c.urlMap.SubclusterToIndex(clusterID, docIndex)
		if chunk == retrievedChunk {
			url := corpus.GetIthUrl(urls, index)

			results = append(results, SearchResult{
				URL:       url,
				Score:     float64(score),
				ClusterID: clusterID,
				Rank:      i + 1, // Will be recomputed later when all clusters are combined
			})
		}
	}

	return results, nil
}

// Safe URL reconstruction that handles errors gracefully
func (c *Client) safeReconstructUrl(clusterID, docIndex uint64, coordinatorAddr string) (string, float64) {
	var totalCommMB float64 = 0.0

	// Try to get URL with error handling
	urlQuery, retrievedChunk := c.QueryUrls(clusterID, docIndex)

	networkingStart := time.Now()
	urlAns := c.getUrlsAnswer(urlQuery, true, coordinatorAddr)

	// Track URL communication regardless of success
	_, uploadMB, downloadMB := logStats(c.NumDocs(), networkingStart, urlQuery, urlAns)
	totalCommMB += uploadMB + downloadMB

	// Try multiple approaches to get a valid URL
	url := c.tryReconstructUrlWithFallbacks(clusterID, docIndex, urlAns, retrievedChunk)

	return url, totalCommMB
}

// Try URL reconstruction with multiple fallback strategies
func (c *Client) tryReconstructUrlWithFallbacks(clusterID, docIndex uint64, urlAns *pir.Answer[matrix.Elem32], retrievedChunk uint64) string {
	var url string

	// Strategy 1: Try original URL reconstruction with panic recovery
	func() {
		defer func() {
			if r := recover(); r != nil {
				// Ignore panic and try fallback
			}
		}()

		urls := c.ReconstructUrls(urlAns, clusterID, docIndex)
		if urls != "" {
			// Extract the specific URL for this document
			_, chunk, index := c.urlMap.SubclusterToIndex(clusterID, docIndex)
			if chunk == retrievedChunk {
				extractedUrl := corpus.GetIthUrl(urls, index)
				if extractedUrl != "" {
					url = extractedUrl
					return
				}
			}
		}
	}()

	// Strategy 2: Try simplified URL reconstruction
	if url == "" {
		url = c.trySimplifiedUrlReconstruction(urlAns, clusterID, docIndex)
	}

	// Strategy 3: Generate a document identifier URL for relevance testing
	if url == "" {
		// Create a searchable document identifier that can be matched against qrels
		// This format allows you to extract the cluster and document for relevance scoring
		url = fmt.Sprintf("tiptoe://cluster-%d/doc-%d", clusterID, docIndex)
	}

	return url
}

// Simplified URL reconstruction that bypasses compression issues
func (c *Client) trySimplifiedUrlReconstruction(answer *pir.Answer[matrix.Elem32], clusterID, docIndex uint64) string {
	defer func() {
		if r := recover(); r != nil {
			// Return empty string if this also fails
		}
	}()

	dbIndex, _, _ := c.urlMap.SubclusterToIndex(clusterID, docIndex)
	rowStart, _ := database.Decompose(dbIndex, c.urlInfo.M)

	// Try a smaller range to avoid compression issues
	maxBytes := min(c.params.UrlBytes, 100) // Limit to 100 bytes to avoid corruption
	rowEnd := rowStart + maxBytes

	vals := c.urlClient.Recover(answer)

	if rowEnd > uint64(len(vals)) {
		rowEnd = uint64(len(vals))
	}

	if rowStart >= rowEnd {
		return ""
	}

	out := make([]byte, rowEnd-rowStart)
	for i, e := range vals[rowStart:rowEnd] {
		out[i] = byte(e)
	}

	// Try without decompression first
	result := strings.TrimRight(string(out), "\x00")

	// Basic validation - check if it looks like a URL
	if strings.Contains(result, "http") || strings.Contains(result, "www") {
		return result
	}

	return ""
}

func min(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

// func (c *Client) runRound(p Perf, in io.WriteCloser, out io.ReadCloser,
// 	text, coordinatorAddr string, verbose, keepConn bool) Perf {
// 	y := color.New(color.FgYellow, color.Bold)
// 	fmt.Printf("Executing query \"%s\"\n", y.Sprintf(text))

// 	// Build embeddings query
// 	start := time.Now()
// 	if verbose {
// 		c.printStep("Generating embedding of the query")
// 	}

// 	var query struct {
// 		Cluster_index uint64
// 		Emb           []int8
// 	}

// 	fmt.Printf("DEBUG: about to run python subprotocol, '%s'\n", text)

// 	// Send query
// 	_, err := io.WriteString(in, text+"\n") // send query to embedding process
// 	if err != nil {
// 		fmt.Printf("Debug: error writing to python process: %v\n", err)
// 	}

// 	fmt.Printf("DEBUG: Query sent, waiting for response...\n")

// 	decoder := json.NewDecoder(out)
// 	if err := decoder.Decode(&query); err != nil {
// 		fmt.Printf("DEBUG: Error decoding JSON from Python: %v\n", err)
// 		fmt.Printf("DEBUG: This usually means the Python process crashed or returned invalid JSON\n")
// 		log.Printf("Did you remember to set up your python venv?")
// 		panic(err)
// 	}

// 	fmt.Printf("DEBUG: Successfully received response: cluster=%d, emb_len=%d\n",
// 		query.Cluster_index, len(query.Emb))

// 	// if err := json.NewDecoder(out).Decode(&query); err != nil { // get back embedding + cluster
// 	// 	log.Printf("Did you remember to set up your python venv?")
// 	// 	panic(err)
// 	// }

// 	if query.Cluster_index >= uint64(c.NumClusters()) {
// 		panic("Should not happen")
// 	}

// 	if verbose {
// 		c.printStep(fmt.Sprintf("Building PIR query for cluster %d", query.Cluster_index))
// 	}

// 	embQuery := c.QueryEmbeddings(query.Emb, query.Cluster_index)
// 	p.clientSetup = time.Since(start).Seconds()

// 	// Send embeddings query to server
// 	if verbose {
// 		c.printStep("Sending SimplePIR query to server")
// 	}
// 	networkingStart := time.Now()
// 	embAns := c.getEmbeddingsAnswer(embQuery, true /* keep conn */, coordinatorAddr)
// 	p.t1, p.up1, p.down1 = logStats(c.params.NumDocs, networkingStart, embQuery, embAns)

// 	// Recover document and URL chunk to query for
// 	c.printStep("Decrypting server answer")
// 	embDec := c.ReconstructEmbeddingsWithinCluster(embAns, query.Cluster_index)
// 	scores := embeddings.SmoothResults(embDec, c.embInfo.P())
// 	indicesByScore := utils.SortByScores(scores)
// 	docIndex := indicesByScore[0]

// 	if verbose {
// 		fmt.Printf("\tDoc %d within cluster %d has the largest inner product with our query\n",
// 			docIndex, query.Cluster_index)
// 		c.printStep(fmt.Sprintf("Building PIR query for url/title of doc %d in cluster %d",
// 			docIndex, query.Cluster_index))
// 	}

// 	// Build URL query
// 	urlQuery, retrievedChunk := c.QueryUrls(query.Cluster_index, docIndex)

// 	// Send URL query to server
// 	if verbose {
// 		c.printStep(fmt.Sprintf("Sending PIR query to server for chunk %d", retrievedChunk))
// 	}
// 	networkingStart = time.Now()
// 	urlAns := c.getUrlsAnswer(urlQuery, keepConn, coordinatorAddr)
// 	p.t2, p.up2, p.down2 = logStats(c.params.NumDocs, networkingStart, urlQuery, urlAns)

// 	// Recover URLs of top 10 docs in chunk
// 	urls := c.ReconstructUrls(urlAns, query.Cluster_index, docIndex)
// 	if verbose {
// 		c.printStep("Reconstructed PIR answers.")
// 		fmt.Printf("\tThe top 10 retrieved urls are:\n")
// 	}

// 	j := 1
// 	for at := 0; at < len(indicesByScore); at++ {
// 		if scores[at] == 0 {
// 			break
// 		}

// 		doc := indicesByScore[at]
// 		_, chunk, index := c.urlMap.SubclusterToIndex(query.Cluster_index, doc)

// 		if chunk == retrievedChunk {
// 			if verbose {
// 				fmt.Printf("\t% 3d) [score %s] %s\n", j,
// 					color.YellowString(fmt.Sprintf("% 4d", scores[at])),
// 					color.BlueString(corpus.GetIthUrl(urls, index)))
// 			}
// 			j += 1
// 			if j > 10 {
// 				break
// 			}
// 		}
// 	}

// 	p.clientTotal = time.Since(start).Seconds()
// 	fmt.Printf("\tAnswered in:\n\t\t%v (preproc)\n\t\t%v (client)\n\t\t%v (round 1)\n\t\t%v (round 2)\n\t\t%v (total)\n---\n",
// 		p.clientPreproc, p.clientSetup, p.t1, p.t2, p.clientTotal)

// 	return p
// }

func (c *Client) Setup(hint *TiptoeHint) {
	if hint == nil {
		panic("Hint is empty")
	}

	if hint.CParams.NumDocs == 0 {
		panic("Corpus is empty")
	}

	c.params = hint.CParams
	c.embInfo = &hint.EmbeddingsHint.Info
	c.urlInfo = &hint.UrlsHint.Info

	if hint.ServeEmbeddings {
		if hint.EmbeddingsHint.IsEmpty() {
			panic("Embeddings hint is empty")
		}

		c.embClient = utils.NewUnderhoodClient(&hint.EmbeddingsHint)

		c.embMap = hint.EmbeddingsIndexMap
		c.embIndices = make(map[uint64]bool)
		for _, v := range c.embMap {
			c.embIndices[v] = true
		}

		fmt.Printf("\tEmbeddings client: %s\n", utils.PrintParams(c.embInfo))
	}

	if hint.ServeUrls {
		if hint.UrlsHint.IsEmpty() {
			panic("Urls hint is empty")
		}

		c.urlClient = utils.NewUnderhoodClient(&hint.UrlsHint)

		c.urlMap = hint.UrlsIndexMap
		c.urlIndices = make(map[uint64]bool)
		for _, vals := range c.urlMap {
			for _, v := range vals {
				c.urlIndices[v.Index()] = true
			}
		}

		fmt.Printf("\tURL client: %s\n", utils.PrintParams(c.urlInfo))
	}

	if hint.ServeUrls && hint.ServeEmbeddings &&
		(len(c.urlMap) != len(c.embMap)) {
		fmt.Printf("Both maps don't have the same length: %d %d\n", len(c.urlMap), len(c.embMap))
		//    panic("Both maps don't have same length.")
	}
}

func (c *Client) PreprocessQuery() *underhood.HintQuery {
	if c.params.NumDocs == 0 {
		panic("Not set up")
	}

	if c.embClient != nil {
		hintQuery := c.embClient.HintQuery()
		if c.urlClient != nil {
			c.urlClient.CopySecret(c.embClient)
		}
		return hintQuery
	} else if c.urlClient != nil {
		return c.urlClient.HintQuery()
	} else {
		panic("Should not happen, urlClient is nil")
	}
}

func (c *Client) ProcessHintApply(ans *UnderhoodAnswer) {
	if c.embClient != nil {
		c.embClient.HintRecover(&ans.EmbAnswer)
		c.embClient.PreprocessQueryLHE()
	}

	if c.urlClient != nil {
		c.urlClient.HintRecover(&ans.UrlAnswer)
		c.urlClient.PreprocessQuery()
	}
}

func (c *Client) QueryEmbeddings(emb []int8, clusterIndex uint64) *pir.Query[matrix.Elem64] {
	if c.params.NumDocs == 0 {
		panic("Not set up")
	}

	dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
	m := c.embInfo.M
	dim := uint64(len(emb))

	if m%dim != 0 {
		panic("Should not happen, m should be multiple of dim")
	}
	if dbIndex%dim != 0 {
		panic("Should not happen, dbIndex should be multiple of dim")
	}

	_, colIndex := database.Decompose(dbIndex, m)
	arr := matrix.Zeros[matrix.Elem64](m, 1)
	for j := uint64(0); j < dim; j++ {
		arr.AddAt(colIndex+j, 0, matrix.Elem64(emb[j]))
	}

	return c.embClient.QueryLHE(arr)
}

func (c *Client) QueryUrls(clusterIndex, docIndex uint64) (*pir.Query[matrix.Elem32], uint64) {
	if c.params.NumDocs == 0 {
		panic("Not set up")
	}

	dbIndex, chunkIndex, _ := c.urlMap.SubclusterToIndex(clusterIndex, docIndex)

	return c.urlClient.Query(dbIndex), chunkIndex
}

func (c *Client) ReconstructEmbeddings(answer *pir.Answer[matrix.Elem64],
	clusterIndex uint64) uint64 {
	vals := c.embClient.RecoverLHE(answer)

	dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
	rowIndex, _ := database.Decompose(dbIndex, c.embInfo.M)
	res := vals.Get(rowIndex, 0)

	return uint64(res)
}

func (c *Client) ReconstructEmbeddingsWithinCluster(answer *pir.Answer[matrix.Elem64],
	clusterIndex uint64) []uint64 {
	dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
	rowStart, colIndex := database.Decompose(dbIndex, c.embInfo.M)
	rowEnd := database.FindEnd(c.embIndices, rowStart, colIndex,
		c.embInfo.M, c.embInfo.L, 0)

	vals := c.embClient.RecoverLHE(answer)

	res := make([]uint64, rowEnd-rowStart)
	at := 0
	for j := rowStart; j < rowEnd; j++ {
		res[at] = uint64(vals.Get(j, 0))
		at += 1
	}

	return res
}

func (c *Client) ReconstructUrls(answer *pir.Answer[matrix.Elem32],
	clusterIndex, docIndex uint64) string {
	dbIndex, _, _ := c.urlMap.SubclusterToIndex(clusterIndex, docIndex)
	rowStart, colIndex := database.Decompose(dbIndex, c.urlInfo.M)
	rowEnd := database.FindEnd(c.urlIndices, rowStart, colIndex,
		c.urlInfo.M, c.urlInfo.L, c.params.UrlBytes)

	vals := c.urlClient.Recover(answer)

	out := make([]byte, rowEnd-rowStart)
	for i, e := range vals[rowStart:rowEnd] {
		out[i] = byte(e)
	}

	if c.params.CompressUrl {
		res, err := corpus.Decompress(out)
		for err != nil {
			out = out[:len(out)-1]
			if len(out) == 0 {
				panic("Should not happen, url is empty")
			}
			res, err = corpus.Decompress(out)
		}
		return strings.TrimRight(res, "\x00")
	}

	return strings.TrimRight(string(out), "\x00")
}

func makeRPC[Q QueryType, A AnsType](query *Q, reply *A, useCoordinator, keepConn bool,
	tcp, rpc string, client *rpc.Client) *rpc.Client {
	if !useCoordinator {
		conn := utils.DialTCP(tcp)
		utils.CallTCP(conn, "Server."+rpc, query, reply)
		conn.Close()
	} else {
		if client == nil {
			client = utils.DialTLS(tcp)
		}

		utils.CallTLS(client, "Coordinator."+rpc, query, reply)

		if !keepConn {
			client.Close()
			client = nil
		}
	}

	return client
}

func (c *Client) getHint(keepConn bool, tcp string) *TiptoeHint {
	query := true
	hint := TiptoeHint{}
	c.rpcClient = makeRPC[bool, TiptoeHint](&query, &hint, c.useCoordinator, keepConn,
		tcp, "GetHint", c.rpcClient)
	return &hint
}

func (c *Client) applyHint(ct *underhood.HintQuery,
	keepConn bool,
	tcp string) *UnderhoodAnswer {
	ans := UnderhoodAnswer{}
	c.rpcClient = makeRPC[underhood.HintQuery, UnderhoodAnswer](ct, &ans,
		c.useCoordinator, keepConn,
		tcp, "ApplyHint",
		c.rpcClient)
	return &ans
}

func (c *Client) getEmbeddingsAnswer(query *pir.Query[matrix.Elem64],
	keepConn bool,
	tcp string) *pir.Answer[matrix.Elem64] {
	ans := pir.Answer[matrix.Elem64]{}
	c.rpcClient = makeRPC[pir.Query[matrix.Elem64], pir.Answer[matrix.Elem64]](query, &ans,
		c.useCoordinator, keepConn,
		tcp, "GetEmbeddingsAnswer",
		c.rpcClient)
	return &ans
}

func (c *Client) getUrlsAnswer(query *pir.Query[matrix.Elem32],
	keepConn bool,
	tcp string) *pir.Answer[matrix.Elem32] {
	ans := pir.Answer[matrix.Elem32]{}
	c.rpcClient = makeRPC[pir.Query[matrix.Elem32], pir.Answer[matrix.Elem32]](query, &ans,
		c.useCoordinator, keepConn,
		tcp, "GetUrlsAnswer",
		c.rpcClient)
	return &ans
}

func (c *Client) closeConn() {
	if c.rpcClient != nil {
		c.rpcClient.Close()
		c.rpcClient = nil
	}
}
