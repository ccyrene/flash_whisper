package main

import (
    "fmt"
    "math"
    "strings"
    "encoding/binary"
    "encoding/base64"
)

type Comparable interface {
	~int | ~float64
}

type resultWithIndex struct {
	taskId int
	result string
}

// type TranscriptionRequest struct {
//     Audio           string  `json:"audio"`
//     Language        *string `json:"language,omitempty"`
//     ChunkDuration   *int    `json:"chunk_duration,omitempty"`
//     MaxNewTokens    *int    `json:"max_new_tokens,omitempty"`
//     NumTasks        *int    `json:"num_tasks,omitempty"`
// }

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

// Helper function to read a single int32 from bytes (Little Endian)
func readInt32(data []byte) int32 {
	return int32(binary.LittleEndian.Uint32(data))
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