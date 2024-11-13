package main

import (
    "log"
    "fmt"
    "sync"
    "time"
    "strings"
    "encoding/base64"
    "github.com/gofiber/fiber/v2"
)

// import (
//     "io"
//     "fmt"
//     "time"
//     "bytes"
//     "encoding/base64"
//     "github.com/youpy/go-wav"
//     "github.com/gofiber/fiber/v2"
// )

func healthCheck(c *fiber.Ctx) error {
    return c.SendString("I'm still alive!")
}

func transcribe(c *fiber.Ctx) error {

    startProcess := time.Now()

    startTime := time.Now()
    var jsonPayload map[string]interface{}
    if err := c.BodyParser(&jsonPayload); err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error": "Invalid JSON format",
        })
    }

    elapsed := time.Since(startTime)
    fmt.Printf("Execution time for parseJSON: %d ms\n", elapsed.Milliseconds())

    startTime = time.Now()    
    userInputs, err := getData(jsonPayload)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error": err.Error(),
        })
    }

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for getData: %d ms\n", elapsed.Milliseconds())

    startTime = time.Now()
    audioBytes, err := base64.StdEncoding.DecodeString(userInputs["audio"].(string))
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error": "Failed to decode audio",
        })
    }

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for decodeBase64: %d ms\n", elapsed.Milliseconds())
    // // Before loading audio
    // fmt.Printf("Loaded audio bytes length: %d\n", len(audioBytes))

    startTime = time.Now()
    sampleRate, numChannels, err := parseWAVHeader(audioBytes)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error":   "Error parsing WAV header",
            "details": err.Error(),
        })
    }

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for parseWAVHeader: %d ms\n", elapsed.Milliseconds())
    fmt.Printf("Sample Rate: %d, Channels: %d\n", sampleRate, numChannels)

    startTime = time.Now()
    wavData, err := readWAVData(audioBytes, numChannels)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error":   "Error reading WAV data",
            "details": err.Error(),
        })
    }

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for readWAVData: %d ms\n", elapsed.Milliseconds())
    fmt.Printf("WAV Length: %d, sampleRate: %d\n", len(wavData), sampleRate)

    if sampleRate != 16000 {

        startTime = time.Now()

        wavData, err = resample(wavData, sampleRate, 16000)
        if err != nil {
            return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
        }

        elapsed = time.Since(startTime)
        fmt.Printf("Execution time for resample: %d ms\n", elapsed.Milliseconds())
    }

    startTime = time.Now()
    batchAudio, err := processLargeAudio(wavData, userInputs["chunk_duration"].(int))
    if err != nil {
        return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
    }

    numTasks := userInputs["num_tasks"].(int)
    dpsList, err := splitData(batchAudio, numTasks)
    if err != nil {
        return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
    }

    numTasks = min(numTasks, len(dpsList))

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for cal&split chunks: %d ms\n", elapsed.Milliseconds())

    startTime = time.Now()
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

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for inference: %d ms\n", elapsed.Milliseconds())

    startTime = time.Now()
	// Collect results and sort them by taskId
	var results []string
	resultMap := make(map[int][]string) // To store results by taskId

	// Process each result, storing them by taskId
	for result := range resultChan {
		resultMap[result.taskId] = append(resultMap[result.taskId], result.result)
	}

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for map results: %d ms\n", elapsed.Milliseconds())

    startTime = time.Now()
	// Now that results are grouped by taskId, sort them by taskId order
	for i := 0; i < numTasks; i++ {
		// Process the results for each task in order
		for _, res := range resultMap[i] {
			results = append(results, res)
		}
	}

    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for sort results: %d ms\n", elapsed.Milliseconds())

    startTime = time.Now()
    res := postprocessString(results)
    elapsed = time.Since(startTime)
    fmt.Printf("Execution time for postprocessString: %d ms\n", elapsed.Milliseconds())

    elapsed = time.Since(startProcess)
    fmt.Printf("Execution time for allProcess: %d ms\n", elapsed.Milliseconds())

    fmt.Println(strings.Repeat("=", 100))

    return c.JSON(fiber.Map{
        "text": res,
        "wavData": wavData,
    })

}