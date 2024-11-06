package main

import (
	"log"
	"fmt"
	"time"
	"strconv"
	"context"
	"encoding/binary"
	triton "goclient/grpc-client"
)

func ServerLiveRequest(client triton.GRPCInferenceServiceClient) *triton.ServerLiveResponse {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverLiveRequest := triton.ServerLiveRequest{}
	serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)
	if err !=  nil {
		log.Fatalf("Couldn't get server live: %v", err)
	}
	return serverLiveResponse
}

func ServerReadyRequest(client triton.GRPCInferenceServiceClient) *triton.ServerReadyResponse {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverReadyRequest := triton.ServerReadyRequest{}
	serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		log.Fatalf("Couldn't get server ready: %v", err)
	}
	return serverReadyResponse
}

func ModelMetadataRequest(client triton.GRPCInferenceServiceClient, modelName string, modelVersion string) *triton.ModelMetadataResponse {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	modelMetadataRequest := triton.ModelMetadataRequest{
		Name: modelName,
		Version: modelVersion,
	}

	modelMetadataResponse, err := client.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		log.Fatalf("Couldn't get server model meta data: %v", err)
	}
	return modelMetadataResponse
}

func ModelInferRequest(client triton.GRPCInferenceServiceClient, wav []byte, wavLen []byte, maxNewTokens []byte, prompt []byte) *triton.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "WAV",
			Datatype: "FP32",
			Shape:    []int64{1, 480000},
		},
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "WAV_LENS",
			Datatype: "INT32",
			Shape:    []int64{1, 1},
		},
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "MAX_NEW_TOKENS",
			Datatype: "INT32",
			Shape:    []int64{1, 1},
		},
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "TEXT_PREFIX",
			Datatype: "STRING",
			Shape:    []int64{1, 1},
		},
	}

	// Create request input output tensors
	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "TRANSCRIPTS",
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := triton.ModelInferRequest{
		ModelName:    "infer_bls",
		ModelVersion: "",
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, wav)
	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, wavLen)
	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, maxNewTokens)
	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, prompt)

	// Submit inference request to server
	modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return modelInferResponse
}

// Helper function to read a single int32 from bytes (Little Endian)
func readInt32(data []byte) int32 {
	return int32(binary.LittleEndian.Uint32(data))
}

func preprocessString(inputStrList []string, batchSize int) []byte {

    if batchSize > len(inputStrList) {
        batchSize = len(inputStrList)
    }

    var inputStrBytes []byte
    totalSize := 0
    for b := 0; b < batchSize; b++ {
        totalSize += 4 + len(inputStrList[b]) // 4 bytes for length + string bytes
    }
    inputStrBytes = make([]byte, 0, totalSize)

    bs := make([]byte, 4) // To hold length as little-endian 4 bytes
    for b := 0; b < batchSize; b++ {
        inputStr := inputStrList[b]
        strBytes := []byte(inputStr)
        strCap := len(inputStr)
        binary.LittleEndian.PutUint32(bs, uint32(strCap))
        inputStrBytes = append(inputStrBytes, bs...) // Append length
        inputStrBytes = append(inputStrBytes, strBytes...) // Append string
    }
    return inputStrBytes
}

func sendWhisper(
	client triton.GRPCInferenceServiceClient,
	dps [][]float32,
	name string,
	maxNewTokens int32,
	language string,
) ([]string, error) {

	results := []string{}
	// Convert substring to integer
	taskId, err := strconv.Atoi(name[5:])
	if err != nil {
		return nil, fmt.Errorf("Error converting substring to integer: %w", err)
	}

	for i, dp := range dps {
		// Prepare WAV tensor
		samples := make([]float32, 480000)
		copy(samples, dp)

		// Prepare length tensor
		wavLen := []int32{int32(len(dp))}

		// Prepare TEXT_PREFIX and MAX_NEW_TOKENS tensors
		promptList := []string{fmt.Sprintf("<|startoftranscript|><|%s|><|transcribe|><|notimestamps|>", language)}
		prompt := preprocessString(promptList, 1)

		maxTokens := []int32{maxNewTokens}

		// Set up input tensors
		inferInputs := []*triton.ModelInferRequest_InferInputTensor{
			{
				Name:     "WAV",
				Datatype: "FP32",
				Shape:    []int64{1, 480000},
			},
			{
				Name:     "WAV_LENS",
				Datatype: "INT32",
				Shape:    []int64{1, 1},
			},
			{
				Name:     "TEXT_PREFIX",
				Datatype: "BYTES",
				Shape:    []int64{1, 1},
			},
			{
				Name:     "MAX_NEW_TOKENS",
				Datatype: "INT32",
				Shape:    []int64{1, 1},
			},
		}

		// Set up output tensor
		inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
			{Name: "TRANSCRIPTS"},
		}

		// Create ModelInferRequest
		sequenceId := 100000000 + i + taskId * 10
		modelInferRequest := &triton.ModelInferRequest{
			ModelName:    "infer_bls",
			ModelVersion: "1",
			Id: 		  strconv.Itoa(sequenceId),
			Inputs:       inferInputs,
			Outputs:      inferOutputs,
			RawInputContents: [][]byte{
				float32ToByte(samples),
				int32ToByte(wavLen),
				prompt,
				int32ToByte(maxTokens),
			},
		}

		// Set up a context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
		defer cancel()

		// Perform inference
		response, err := client.ModelInfer(ctx, modelInferRequest)
		if err != nil {
			return nil, fmt.Errorf("error during inference: %w", err)
		}

		outputBytes := response.RawOutputContents[0]
		// fmt.Println(strings.Repeat("=", 100))
		// fmt.Println(i)
		// fmt.Println(outputBytes)
		// fmt.Println(strings.Repeat("=", 100))
		transcripts := string(outputBytes[4:])

		// Append each transcript to the result slice
		results = append(results, transcripts)
	}

	return results, nil
}