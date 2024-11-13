package main

import (
    "io"
    "fmt"
    "bytes"
    "github.com/youpy/go-wav"
)

func getData(request map[string]interface{}) (map[string]interface{}, error) {
    userInputs := make(map[string]interface{})
    requiredParams := []string{"audio"}
    defaultParams := []string{"language", "chunk_duration", "max_new_tokens", "num_tasks"}

    for _, param := range requiredParams {
        if val, exists := request[param]; exists {
            validatedValue, err := validateRequest(val, param)
            if err != nil {
                return nil, err
            }
            userInputs[param] = validatedValue
        } else {
            return nil, fmt.Errorf("'%s' parameter is required", param)
        }
    }

    for _, param := range defaultParams {
        if val, exists := request[param]; exists && val != nil {
            validatedValue, err := validateRequest(val, param)
            if err != nil {
                return nil, err
            }
            userInputs[param] = validatedValue
        } else {
            userInputs[param] = setDefault(param)
        }
    }
    
    return userInputs, nil
}

// Simple linear interpolation resampling function
func resample(samples []float32, srcRate int, targetRate int) ([]float32, error) {
    if targetRate == srcRate {
        return samples, nil
    }

    ratio := float64(srcRate) / float64(targetRate)
    newLength := int(float64(len(samples)) / ratio)
    resampled := make([]float32, newLength)

    for i := 0; i < newLength; i++ {
        srcIndex := float64(i) * ratio
        srcIndexInt := int(srcIndex)
        if srcIndexInt+1 < len(samples) {
            nextSample := samples[srcIndexInt+1]
            resampled[i] = samples[srcIndexInt] + float32(srcIndex-float64(srcIndexInt))*(nextSample-samples[srcIndexInt])
        } else {
            resampled[i] = samples[srcIndexInt]
        }
    }

    return resampled, nil
}

func readWAVData(bpayload []byte, numChannels int) ([]float32, error) {

    reader := wav.NewReader(bytes.NewReader(bpayload))

    var wavData []float32

    for {
        samples, err := reader.ReadSamples()
        if err == io.EOF {
            break
        }

        for _, sample := range samples {
            var sampleValue float64
            if numChannels == 2 {
                sampleValue = (reader.FloatValue(sample, 0) + reader.FloatValue(sample, 1)) / 2
            } else {
                sampleValue = reader.FloatValue(sample, 0)
            }
            wavData = append(wavData, float32(sampleValue))
        }
    }

    return wavData, nil
}

func processLargeAudio(audio []float32, chunkDuration int) ([][]float32, error) {
	// Calculate the number of samples per chunk
	samplesPerChunk := chunkDuration * 16000
	totalSamples := len(audio)
	// Calculate the number of chunks, rounding up if there's a remainder
	numChunks := (totalSamples + samplesPerChunk - 1) / samplesPerChunk

	// Initialize the 2D slice to hold audio chunks
	batchAudio := make([][]float32, 0)

	for i := 0; i < numChunks; i++ {
		// Calculate start and end indices for each chunk
		start := i * samplesPerChunk
		end := start + samplesPerChunk
		if end > totalSamples {
			end = totalSamples // Adjust the last chunk if it exceeds total samples
		}

		// Slice the audio for the current chunk and append to batchAudio
		batchAudio = append(batchAudio, audio[start:end])
	}

	return batchAudio, nil
}

func splitData(data [][]float32, k int) ([][][]float32, error) {
	n := len(data)
	if n < k {
		fmt.Printf("Warning: the length of the input list (%d) is less than k (%d). Setting k to %d.\n", n, k, n)
		k = n
	} else {
		fmt.Printf("has %d chunks.\n", n)
	}

	quotient := n / k
	remainder := n % k

	var result [][][]float32
	start := 0
	for i := 0; i < k; i++ {
		end := start + quotient
		if i < remainder {
			end++
		}

		result = append(result, data[start:end])
		start = end
	}

	return result, nil
}