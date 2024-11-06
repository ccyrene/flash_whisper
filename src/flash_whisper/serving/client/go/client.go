package main

import (
	// "os"
	"fmt"
	"log"
	"flag"
	"runtime"

	"github.com/joho/godotenv"
	"github.com/gofiber/fiber/v2"

    "google.golang.org/grpc"
	triton "goclient/grpc-client"
)

var tritonClient triton.GRPCInferenceServiceClient

type Config struct {
    Host    string
    Port    string
    Workers int
}

func parseFlags() Config {
    host := flag.String("host", "", "Server host")
    port := flag.String("port", "8080", "Server port")
    workers := flag.Int("workers", 4, "Number of CPU cores to use")
    flag.Parse()

    return Config{
        Host:    *host,
        Port:    *port,
        Workers: *workers,
    }
}

func main() {

    config := parseFlags()
    runtime.GOMAXPROCS(config.Workers)

    log.SetFlags(log.LstdFlags | log.Lshortfile)

    if err := godotenv.Load(); err != nil {
        log.Println("No .env file found, proceeding with default environment")
    }

    conn, err := grpc.Dial("localhost:8001", grpc.WithInsecure())
    // conn, err := grpc.Dial(os.Getenv("TRITON_SERVER_ENDPOINT"), grpc.WithInsecure())
    if err != nil {
        log.Fatalf("Couldn't connect to endpoint %s: %v", config.Host, err)
    }

	defer conn.Close()

	// Create client from gRPC server connection
	tritonClient = triton.NewGRPCInferenceServiceClient(conn)

	serverLiveResponse := ServerLiveRequest(tritonClient)
	fmt.Printf("Triton Health - Live: %v\n", serverLiveResponse.Live)

	serverReadyResponse := ServerReadyRequest(tritonClient)
	fmt.Printf("Triton Health - Ready: %v\n", serverReadyResponse.Ready)

	modelMetadataResponse := ModelMetadataRequest(tritonClient, "infer_bls", "")
	fmt.Println(modelMetadataResponse)

    app := fiber.New(fiber.Config{
        Prefork: true,
        BodyLimit: 3 * 1024 * 1024 * 1024,
    })

	app.Get("/", healthCheck)
    app.Post("/transcribe", transcribe)

    address := fmt.Sprintf("%s:%s", config.Host, config.Port)
    log.Printf("Starting server on %s with %d CPU core(s)", address, config.Workers)

    if err := app.Listen(address); err != nil {
        log.Fatalf("Server failed to start: %v", err)
    }

}