package main

import (
	"log"
	"fmt"
	"time"
	"strconv"
	"context"
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
		wavLenInt := len(dp)

		// Prepare WAV tensor
		samples := make([]float32, wavLenInt)
		copy(samples, dp)

		// Prepare length tensor
		wavLen := []int32{int32(wavLenInt)}

		// Prepare TEXT_PREFIX and MAX_NEW_TOKENS tensors
		promptList := []string{fmt.Sprintf("<|startoftranscript|><|%s|><|transcribe|><|notimestamps|>", language)}
		prompt := preprocessString(promptList, 1)

		maxTokens := []int32{maxNewTokens}

		// Set up input tensors
		inferInputs := []*triton.ModelInferRequest_InferInputTensor{
			{
				Name:     "WAV",
				Datatype: "FP32",
				Shape:    []int64{1, wavLenInt},
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
		transcripts := string(outputBytes[4:])
		results = append(results, transcripts)
	}

	return results, nil
}