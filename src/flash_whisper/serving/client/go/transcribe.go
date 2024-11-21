package main

import (
    "log"
    "fmt"
    "sync"
    "encoding/base64"
    "github.com/gofiber/fiber/v2"
)

func healthCheck(c *fiber.Ctx) error {
    return c.SendString("I'm still alive!")
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

    sampleRate, numChannels, err := parseWAVHeader(audioBytes)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error":   "Error parsing WAV header",
            "details": err.Error(),
        })
    }

    wavData, err := readWAVData(audioBytes, numChannels)
    if err != nil {
        return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
            "error":   "Error reading WAV data",
            "details": err.Error(),
        })
    }

    if sampleRate != 16000 {
        wavData, err = resample(wavData, sampleRate, 16000)
        if err != nil {
            return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
        }
    }

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