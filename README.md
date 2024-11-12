# Flash Whisper: ASR system for real-time application

## Quick Start

```bash
$git clone https://github.com/rungrodkspeed/flash_whisper

$cd flash_whisper/src/flash_whisper/serving

$docker-compose up --build .
```

**required space about 37-38 GiB

## Optimization Solution

<image src="/media/solution.jpg"/>


## Performance

Testing on :
 - CPU: AMD Ryzen 7 5800H with Radeon Graphics 3.20 GHz Processor
 - GPU: NVIDIA GeForce RTX 3060 Laptop GPU

### Server side
** Result of experiment using Whisper-tiny because my RAM not enough (in case using model in CPU) for using whisper-medium or more size.
|Framework | RTF (⭣) | Throughput (⭡) |
| ------------- | ------------- | ------------- |
| Transformers (CPU) |  0.55  | 1.82 |
| ONNX (CPU) |  0.0099  | 101.32 |
| Transformers (GPU) |  0.041  | 24.39 |
| TensorRT-LLM (GPU) |  0.0008 | 1250 |
<image src="/media/rtf.png"/>
<image src="/media/throughput.png"/>

### Client side
** Fix Server-side is Whisper-medium converting by TensorRT-LLM which deploy on Triton Inference Server.
| Audio Length (min) | Python (FastAPI)  | GO (API) |
| ------------- | ------------- | ------------- |
| 0.5 |  434.46 ms  | 259 ms |
| 1 |  711.39 ms  | 656 ms |
| 5 |  12.474 sec  | 8.169 sec |
| 10 |  26.208 sec  | 15.698 sec |
| 20 |  38.169 sec  | 31.235 sec |
| 30 |  62.582 sec  | 47.805 sec |
| 60 |  122.956 sec  | 102.681 sec |
<image src="/media/end2end.png"/>