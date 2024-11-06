package main

import (
    "io"
    "fmt"
    "bytes"
    "encoding/binary"
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

func readPCMData(bpayload []byte, dataStart int, numChannels int, bitDepth int) ([]float32, error) {
    // Create a reader for PCM data starting after the header
    reader := bytes.NewReader(bpayload[dataStart:])
    var samples []float32

    // Calculate bytes per sample
    bytesPerSample := bitDepth / 8
    frameSize := numChannels * bytesPerSample

    // Read samples until EOF
    for {
        frame := make([]byte, frameSize)
        if _, err := io.ReadFull(reader, frame); err == io.EOF {
            break
        } else if err != nil {
            return nil, fmt.Errorf("failed to read PCM frame: %v", err)
        }

        // If the audio is stereo, combine the two channels into one (mono)
        if numChannels == 2 {
            // Average the left and right channels for each sample frame
            var monoSampleValue float32
            for i := 0; i < 2; i++ {
                // Extract sample data for the channel
                sampleBytes := frame[i*bytesPerSample : (i+1)*bytesPerSample]

                // Convert sample to float32
                var sampleValue float32
                switch bitDepth {
                case 16:
                    sampleValue = float32(int16(binary.LittleEndian.Uint16(sampleBytes))) / 32768.0
                case 24:
                    sampleValue = float32(int32(sampleBytes[0])|(int32(sampleBytes[1])<<8)|(int32(sampleBytes[2])<<16)) / 8388608.0
                default:
                    return nil, fmt.Errorf("unsupported bit depth: %d", bitDepth)
                }

                // Add to the mono sample value
                monoSampleValue += sampleValue
            }

            // Take the average of the two channels and append to samples
            monoSampleValue /= 2
            samples = append(samples, monoSampleValue)
        } else if numChannels == 1 {
            // Handle the case for mono audio (no need to average channels)
            for i := 0; i < numChannels; i++ {
                sampleBytes := frame[i*bytesPerSample : (i+1)*bytesPerSample]

                var sampleValue float32
                switch bitDepth {
                case 16:
                    sampleValue = float32(int16(binary.LittleEndian.Uint16(sampleBytes))) / 32768.0
                case 24:
                    sampleValue = float32(int32(sampleBytes[0])|(int32(sampleBytes[1])<<8)|(int32(sampleBytes[2])<<16)) / 8388608.0
                default:
                    return nil, fmt.Errorf("unsupported bit depth: %d", bitDepth)
                }

                samples = append(samples, sampleValue)
            }
        } else {
            return nil, fmt.Errorf("unsupported number of channels: %d", numChannels)
        }
    }
    return samples, nil
}

func processLargeAudio(audio []float32, chunkDuration int) ([][]float32, error) {
	// Calculate the number of samples per chunk
	samplesPerChunk := chunkDuration * 16000
	totalSamples := len(audio)
	// Calculate the number of chunks, rounding up if there's a remainder
	numChunks := (totalSamples + samplesPerChunk - 1) / samplesPerChunk

	// Initialize the 2D slice to hold audio chunks
	batchAudio := make([][]float32, numChunks)

	for i := 0; i < numChunks; i++ {
		// Calculate start and end indices for each chunk
		start := i * samplesPerChunk
		end := start + samplesPerChunk
		if end > totalSamples {
			end = totalSamples // Adjust the last chunk if it exceeds total samples
		}

		// Slice the audio for the current chunk and append to batchAudio
		batchAudio[i] = audio[start:end]
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