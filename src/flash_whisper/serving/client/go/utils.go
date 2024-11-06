package main

import (
    "fmt"
    "log"
    "math"
    "sync"
    "strings"
    "encoding/binary"
    "encoding/base64"
    "github.com/gofiber/fiber/v2"
)

type Comparable interface {
	~int | ~float64
}

type TranscriptionRequest struct {
    Audio           string  `json:"audio"`
    Language        *string `json:"language,omitempty"`
    ChunkDuration   *int    `json:"chunk_duration,omitempty"`
    MaxNewTokens    *int    `json:"max_new_tokens,omitempty"`
    NumTasks        *int    `json:"num_tasks,omitempty"`
}

func validateRequest(param interface{}, paramName string) (interface{}, error) {
    switch paramName {
    case "audio":
        audioStr, ok := param.(string)
        if !ok {
            return nil, fmt.Errorf("Invalid '%s' parameter. Must be a base64 string", paramName)
        }
        _, err := base64.StdEncoding.DecodeString(audioStr)
        if err != nil {
            return nil, fmt.Errorf("Invalid '%s' parameter. Base64 decoding failed", paramName)
        }
        return audioStr, nil

    case "language":
        language, ok := param.(string)
        if !ok {
            return nil, fmt.Errorf("Invalid '%s' parameter. Must be a string", paramName)
        }
        return language, nil

    case "chunk_duration":
        chunkDuration, ok := param.(float64)
        if !ok {
            return nil, fmt.Errorf("Invalid '%s' parameter. Must be an integer", paramName)
        }
        return int(max(5, min(chunkDuration, 30))), nil

    case "max_new_tokens":
        maxNewTokens, ok := param.(float64)
        if !ok {
            return nil, fmt.Errorf("Invalid '%s' parameter. Must be an integer", paramName)
        }
        return int(max(2, min(maxNewTokens, 430))), nil

    case "num_tasks":
        numTasks, ok := param.(float64)
        if !ok {
            return nil, fmt.Errorf("Invalid '%s' parameter. Must be an integer", paramName)
        }
        return int(max(1, numTasks)), nil
    }
    return nil, nil
}

func setDefault(paramName string) interface{} {
    defaultParams := map[string]interface{}{
        "language":       "en",
        "chunk_duration":  30,
        "max_new_tokens":  96,
        "num_tasks":       50,
    }
    return defaultParams[paramName]
}

func max[T Comparable](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func min[T Comparable](a, b T) T {
	if a < b {
		return a
	}
	return b
}

func int32ToByte(data []int32) []byte {
    byteArray := make([]byte, len(data)*4)
    for i, v := range data {
        byteArray[i*4+0] = byte(v)         // Least significant byte
        byteArray[i*4+1] = byte(v >> 8)    // 2nd least significant byte
        byteArray[i*4+2] = byte(v >> 16)   // 3rd least significant byte
        byteArray[i*4+3] = byte(v >> 24)   // Most significant byte
    }
    return byteArray
}

func float32ToByte(data []float32) []byte {
    byteArray := make([]byte, len(data)*4)
    for i, v := range data {
        bits := math.Float32bits(v)
        binary.LittleEndian.PutUint32(byteArray[i*4:], bits)
    }
    return byteArray
}

func postprocessString(res interface{}) string {
	switch v := res.(type) {
	case string:
		return v
	case []string:
		return strings.Join(v, "\n")
	case [][]string:
		var result []string
		for _, sublist := range v {
			result = append(result, postprocessString(sublist))
		}
		return strings.Join(result, "\n")
	default:
		return ""
	}
}

func healthCheck(c *fiber.Ctx) error {
    return c.SendString("I'm still alive!")
}

type resultWithIndex struct {
	taskId int
	result string
}

func transcribe(c *fiber.Ctx) error {
    var jsonPayload map[string]interface{}
    if err := c.BodyParser(&jsonPayload); err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error": "Invalid JSON format",
        })
    }

    userInputs, err := getData(jsonPayload)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error": err.Error(),
        })
    }

    audioBytes, err := base64.StdEncoding.DecodeString(userInputs["audio"].(string))
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error": "Failed to decode audio",
        })
    }

    // Before loading audio
    fmt.Printf("Loaded audio bytes length: %d\n", len(audioBytes))


    sampleRate, numChannels, bitDepth, dataStart, err := parseWAVHeader(audioBytes)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error":   "Error parsing WAV header",
            "details": err.Error(),
        })
    }
    
    fmt.Printf("Sample Rate: %d, Channels: %d, Bit Depth: %d\n", sampleRate, numChannels, bitDepth)

    pcmData, err := readPCMData(audioBytes, dataStart, numChannels, bitDepth)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error":   "Error reading PCM data",
            "details": err.Error(),
        })
    }

    fmt.Printf("PCM Data: %d\n", len(pcmData))

    resampled, err := resample(pcmData, sampleRate, 16000)
    if err != nil {
        return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
    }

    batchAudio, err := processLargeAudio(resampled, userInputs["chunk_duration"].(int))
    if err != nil {
        return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
    }
    numTasks := userInputs["num_tasks"].(int)
    dpsList, err := splitData(batchAudio, numTasks)
    if err != nil {
        return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
    }

    numTasks = min(numTasks, len(dpsList))

    // var wg sync.WaitGroup
    // resultChan := make(chan string, numTasks)

    // for i:=0; i<numTasks; i++ {
    //     wg.Add(1)
    //     go func(i int) {
    //         defer wg.Done()
    //         result, err := sendWhisper(
    //             tritonClient, 
    //             dpsList[i], 
    //             fmt.Sprintf("task-%d", i), 
    //             int32(userInputs["max_new_tokens"].(int)), 
    //             userInputs["language"].(string),
    //         )
	// 		if err != nil {
	// 			// Handle error appropriately
	// 			log.Printf("Error processing task-%d: %v", i, err)
	// 			resultChan <- fmt.Sprintf("Error processing task-%d", i)
	// 			return
	// 		}
    //         for _, res := range result {
    //             resultChan <- res
    //         }
    //     }(i)
    // }

    // go func(){
    //     wg.Wait()
    //     close(resultChan)
    // }()

	// var results []string
	// for result := range resultChan {
	// 	results = append(results, result)
	// }

	var wg sync.WaitGroup
	resultChan := make(chan resultWithIndex, numTasks)

	// Simulate task processing using goroutines
	for i := 0; i < numTasks; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Simulating sendWhisper, replace it with your actual function call
			result, err := sendWhisper(
                            tritonClient, 
                            dpsList[i], 
                            fmt.Sprintf("task-%d", i), 
                            int32(userInputs["max_new_tokens"].(int)), 
                            userInputs["language"].(string),
                        )
			if err != nil {
				log.Printf("Error processing task-%d: %v", i, err)
				resultChan <- resultWithIndex{taskId: i, result: fmt.Sprintf("Error processing task-%d", i)}
				return
			}
			// Send each result with its index
			for _, res := range result {
				resultChan <- resultWithIndex{taskId: i, result: res}
			}
		}(i)
	}

	// Close the channel when all goroutines have finished
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results and sort them by taskId
	var results []string
	resultMap := make(map[int][]string) // To store results by taskId

	// Process each result, storing them by taskId
	for result := range resultChan {
		resultMap[result.taskId] = append(resultMap[result.taskId], result.result)
	}

	// Now that results are grouped by taskId, sort them by taskId order
	for i := 0; i < numTasks; i++ {
		// Process the results for each task in order
		for _, res := range resultMap[i] {
			results = append(results, res)
		}
	}

    res := postprocessString(results)

    return c.JSON(fiber.Map{
        "text": res,
    })
}