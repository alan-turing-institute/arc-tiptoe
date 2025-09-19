package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"strconv"

	"github.com/ahenzinger/tiptoe/search/config"
	"github.com/ahenzinger/tiptoe/search/protocol"
	"github.com/ahenzinger/tiptoe/search/utils"
	"github.com/fatih/color"
)

// Where the corpus is stored
var preamble = flag.String("preamble", "../", "Preamble")

// Path to search config
var searchConfig = flag.String("search_config", "", "Path to search config JSON file")

// Whether or not running image search
var image_search = flag.Bool("image_search", false, "Image search")

func printUsage() {
	fmt.Println("Usage:")
	fmt.Println("\"go run . --preprocess_config path/to/config.json all-servers\" or")
	fmt.Println("\"go run . --preprocess_config path/to/config.json client coordinator-ip\" or")
	fmt.Println("\"go run . --preprocess_config path/to/config.json coordinator numEmbServers numUrlServers ip1 ip2 ...\" or")
	fmt.Println("\"go run . --preprocess_config path/to/config.json emb-server index\" or")
	fmt.Println("\"go run . --preprocess_config path/to/config.json url-server index\"")
	fmt.Println("")
	fmt.Println("Alternative legacy usage:")
	fmt.Println("\"go run . --preamble /path/to/data all-servers\" (uses old directory structure)")
}

func client(coordinatorIP string, conf *config.Config) {
	args := flag.Args()
	if len(args) >= 2 {
		coordinatorIP = args[1]
	}

	protocol.RunClient(utils.RemoteAddr(coordinatorIP, utils.CoordinatorPort), conf)

}

func multiClusterClient(coordinatorIP string, conf *config.Config) {
	args := flag.Args()
	if len(args) >= 2 {
		coordinatorIP = args[1]
	}

	protocol.MultiClusterSearchClient(utils.RemoteAddr(coordinatorIP, utils.CoordinatorPort), conf, true /* verbose */)
}

func multiClusterExperiment(coordinatorIP string, conf *config.Config) {
	args := flag.Args()
	if len(args) >= 2 {
		coordinatorIP = args[1]
	}

	// Run with test query
	col := color.New(color.FgYellow).Add(color.Bold)
	col.Printf("Enter test query: ")
	in := bufio.NewScanner(os.Stdin)
	in.Scan()
	text := in.Text()
	fmt.Printf("\n\n")

	protocol.MultiClusterSearchExperiment(utils.RemoteAddr(coordinatorIP, utils.CoordinatorPort), conf, text, true /* verbose */)
}

func client_latency(coordinatorIP string, conf *config.Config) {
	debug.SetMemoryLimit(20*2 ^ (30))
	args := flag.Args()
	if len(args) >= 2 {
		coordinatorIP = args[1]
	}
	protocol.BenchLatency(101, /* num queries */
		utils.RemoteAddr(coordinatorIP, utils.CoordinatorPort),
		"latency.log", conf)
	utils.WriteFileToStdout("latency.log")
}

func client_tput_embed(coordinatorIP string) {
	args := flag.Args()
	if len(args) >= 2 {
		coordinatorIP = args[1]
	}
	protocol.BenchTputEmbed(utils.RemoteAddr(coordinatorIP, utils.CoordinatorPort),
		"tput_embed.log")
	utils.WriteFileToStdout("tput_embed.log")
}

func client_tput_url(coordinatorIP string) {
	args := flag.Args()
	if len(args) >= 2 {
		coordinatorIP = args[1]
	}
	protocol.BenchTputUrl(utils.RemoteAddr(coordinatorIP, utils.CoordinatorPort),
		"tput_url.log")
	utils.WriteFileToStdout("tput_url.log")
}

func client_tput_offline(coordinatorIP string) {
	args := flag.Args()
	if len(args) >= 2 {
		coordinatorIP = args[1]
	}
	protocol.BenchTputOffline(utils.RemoteAddr(coordinatorIP, utils.CoordinatorPort),
		"tput_offline.log")
	utils.WriteFileToStdout("tput_offline.log")
}

func preprocess_all(conf *config.Config) {
	//debug.SetMemoryLimit(700 * 2^(30))
	protocol.NewEmbeddingServers(0,
		conf.MAX_EMBEDDINGS_SERVERS(),
		conf.EMBEDDINGS_CLUSTERS_PER_SERVER(),
		conf.DEFAULT_EMBEDDINGS_HINT_SZ(),
		true,  // log
		false, // wantCorpus
		false, // serve
		conf)
	fmt.Println("Set up all embedding servers")

	protocol.NewUrlServers(conf.MAX_URL_SERVERS(),
		conf.URL_CLUSTERS_PER_SERVER(),
		config.DEFAULT_URL_HINT_SZ(),
		true,  // log
		false, // wantCorpus
		false, // serve
		conf)
	fmt.Println("Set up all url servers")
}

func preprocess_coordinator(conf *config.Config) {
	debug.SetMemoryLimit(200*2 ^ (30))
	protocol.LocalSetupCoordinator(conf)
}

func all_servers(conf *config.Config) {
	debug.SetMemoryLimit(700*2 ^ (30))
	_, embAddrs, _ := protocol.NewEmbeddingServers(0,
		conf.MAX_EMBEDDINGS_SERVERS(),
		conf.EMBEDDINGS_CLUSTERS_PER_SERVER(),
		conf.DEFAULT_EMBEDDINGS_HINT_SZ(),
		true,  // log
		false, // wantCorpus
		true,  // serve
		conf)
	fmt.Println("Set up all embedding servers")

	_, urlAddrs, _ := protocol.NewUrlServers(conf.MAX_URL_SERVERS(),
		conf.URL_CLUSTERS_PER_SERVER(),
		config.DEFAULT_URL_HINT_SZ(),
		true,  // log
		false, // wantCorpus
		true,  // serve
		conf)
	fmt.Println("Set up all url servers")

	protocol.RunCoordinator(conf.MAX_EMBEDDINGS_SERVERS(),
		conf.MAX_URL_SERVERS(),
		utils.CoordinatorPort,
		append(embAddrs, urlAddrs...),
		true, // log
		conf)
}

func coordinator(conf *config.Config) {
	args := flag.Args()
	numEmbServers, err1 := strconv.Atoi(args[1])
	numUrlServers, err2 := strconv.Atoi(args[2])

	if err1 != nil || err2 != nil || len(args) < 3 {
		panic("Bad input")
	}

	addrs := make([]string, numEmbServers+numUrlServers)
	for i := 0; i < numEmbServers+numUrlServers; i++ {
		ip := "0.0.0.0"
		if i+3 < len(args) {
			ip = args[i+3]
		}

		if i < numEmbServers {
			addrs[i] = utils.RemoteAddr(ip, utils.EmbServerPortStart+i)
		} else {
			addrs[i] = utils.RemoteAddr(ip, utils.UrlServerPortStart+i-numEmbServers)
		}
	}

	protocol.RunCoordinator(numEmbServers,
		numUrlServers,
		utils.CoordinatorPort,
		addrs,
		true, // log
		conf)
}

func build_logs_without_hint(conf *config.Config) {
	debug.SetMemoryLimit(700*2 ^ (30))

	ch := make(chan bool)
	for i := 0; i < conf.MAX_EMBEDDINGS_SERVERS(); i += 20 {
		fmt.Printf("Embedding servers: %d\n", i)
		for j := 0; j < 20; j++ {
			go func(at int) {
				log := conf.EmbeddingServerLog(at)
				logNoHint := conf.EmbeddingServerLogWithoutHint(at)

				if !utils.FileExists(log) {
					panic("Preprocessed cluster server file does not exist")
				}

				server := protocol.NewServerFromFile(log)
				protocol.DumpServerToFileWithoutHint(server, logNoHint)
				ch <- true
			}(i + j)
		}

		for j := 0; j < 20; j++ {
			<-ch
		}
	}

	for i := 0; i < conf.MAX_URL_SERVERS(); i++ {
		go func(at int) {
			fmt.Printf("URL servers: %d\n", at)
			log := conf.UrlServerLog(at)
			logNoHint := conf.UrlServerLogWithoutHint(at)

			if !utils.FileExists(log) {
				panic("Preprocessed url server file does not exist")
			}

			server := protocol.NewServerFromFile(log)
			protocol.DumpServerToFileWithoutHint(server, logNoHint)
			ch <- true
		}(i)
	}

	for i := 0; i < conf.MAX_URL_SERVERS(); i++ {
		<-ch
	}
}

func emb_server(conf *config.Config) {
	args := flag.Args()
	if len(args) < 2 {
		panic("Usage: emb-server index")
	}

	debug.SetMemoryLimit(25*2 ^ (30)) // Necessary so don't run out of memory on r5.xlarge instances
	i, err := strconv.Atoi(args[1])
	if err != nil {
		panic("Invalid server index: " + args[1])
	}

	var log string
	if *image_search {
		log = conf.EmbeddingServerLogWithoutHint(i)
	} else {
		log = conf.EmbeddingServerLog(i)
	}
	if !utils.FileExists(log) {
		panic("Preprocessed cluster server file does not exist")
	}

	var server *protocol.Server
	if *image_search {
		server = protocol.NewServerFromFileWithoutHint(log)
	} else {
		server = protocol.NewServerFromFile(log)
	}
	server.Serve(utils.EmbServerPortStart + i)

}

func url_server(conf *config.Config) {
	args := flag.Args()
	if len(args) < 2 {
		panic("Missing server index argument")
	}

	debug.SetMemoryLimit(25*2 ^ (30)) // Necessary so don't run out of memory on r5.xlarge instances
	i, err := strconv.Atoi(args[1])
	if err != nil {
		panic("Invalid server index")
	}

	var log string
	if *image_search {
		log = conf.UrlServerLogWithoutHint(i)
	} else {
		log = conf.UrlServerLog(i)
	}
	if !utils.FileExists(log) {
		panic("Preprocessed url server file does not exist")
	}

	var server *protocol.Server
	if *image_search {
		server = protocol.NewServerFromFileWithoutHint(log)
	} else {
		server = protocol.NewServerFromFile(log)
	}
	server.Serve(utils.UrlServerPortStart + i)

}

func main() {
	flag.Parse() // Moved to top so flags are parsed before use

	coordinatorIP := "127.0.0.1"

	args := flag.Args()
	if len(args) < 1 { // Using flag.Args() instead of os.Args to avoid parsing issues
		printUsage()
		return
	}

	var conf *config.Config
	var err error

	// Check if search or preprocessing config is provided
	if *searchConfig != "" {
		conf = &config.Config{}
		err = conf.LoadFromSearchConfig(*preamble, *searchConfig)
		if err != nil {
			fmt.Printf("Error loading search config: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Legacy approach - direct preamble
		conf = config.MakeConfig(*preamble+"data", *image_search)
		fmt.Printf("Using legacy data directory: %s\n", conf.PREAMBLE())
	}

	switch args[0] {
	case "client":
		client(coordinatorIP, conf)
	case "multi-cluster-client": // New multi-cluster client
		multiClusterClient(coordinatorIP, conf)
	case "multi-cluster-experiment": // New multi-cluster experiment client
		multiClusterExperiment(coordinatorIP, conf)
	case "client-latency":
		client_latency(coordinatorIP, conf)
	case "client-tput-embed":
		client_tput_embed(coordinatorIP)
	case "client-tput-url":
		client_tput_url(coordinatorIP)
	case "client-tput-offline":
		client_tput_offline(coordinatorIP)
	case "preprocess-all":
		preprocess_all(conf)
	case "preprocess-coordinator":
		preprocess_coordinator(conf)
	case "all-servers":
		all_servers(conf)
	case "coordinator":
		coordinator(conf)
	case "build-logs-without-hint":
		build_logs_without_hint(conf)
	case "emb-server":
		emb_server(conf)
	case "url-server":
		url_server(conf)
	default:
		printUsage()
	}
}
