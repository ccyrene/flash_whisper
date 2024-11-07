package main

import (
    "io"
    "bytes"
    "encoding/binary"
)

func parseWAVHeader(bpayload []byte) (sampleRate int, numChannels int, bitDepth int, dataStart int, err error) {
    reader := bytes.NewReader(bpayload)

    // Skip to channel count (22 bytes in)
    reader.Seek(22, io.SeekStart)

    // Read number of channels (2 bytes)
    var channels uint16
    if err = binary.Read(reader, binary.LittleEndian, &channels); err != nil {
        return
    }
    numChannels = int(channels)

    // Read sample rate (4 bytes)
    var rate uint32
    if err = binary.Read(reader, binary.LittleEndian, &rate); err != nil {
        return
    }
    sampleRate = int(rate)

    // Skip 6 bytes (ByteRate and BlockAlign) to reach `BitsPerSample`
    reader.Seek(6, io.SeekCurrent)

    // Read bit depth (2 bytes)
    var depth uint16
    if err = binary.Read(reader, binary.LittleEndian, &depth); err != nil {
        return
    }
    bitDepth = int(depth)

    // Set starting position of PCM data after the 44-byte header
    dataStart = 44
    return sampleRate, numChannels, bitDepth, dataStart, nil
}