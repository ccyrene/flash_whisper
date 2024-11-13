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

## Get Start
```bash
$git clone https://github.com/rungrodkspeed/flash_whisper
$cd flash_whisper
$pip3 install -e .
```

### ONNX
```bash
$optimum-cli export onnx --optimize=O3 --no-post-process --model=openai/{MODEL_SIZE} --task=automatic-speech-recognition-with-past outs

$python3 inference.py --onnx --model_dir=outs
```

```python
import librosa
from flash_whisper.onnx import ORTWhisper

audio_path = "./sample/sample0.flac"
audio, sr = librosa.load(audio_path)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

model = ORTWhisper(model_dir="outs/")
transcripts = model(audio)
```


### TensorRT-LLM
- Setup
```bash
$apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs
$python3 -m pip install tensorrt-llm==0.15.0.dev2024101500
```

- Preparing material
```bash
$
```

```python
import librosa
from flash_whisper.tllm import WhisperTRTLLM

audio_path = "./sample/sample0.flac"
audio, sr = librosa.load(audio_path)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

engine_dir = "ENGINE_DIR"
model = WhisperTRTLLM(engine_dir, n_mels=80)
transcripts = model(audio)
```

for Windows's OS setup: https://nvidia.github.io/TensorRT-LLM/installation/windows.html \
**If you using Windows I recommend using Wsl(Windows Subsystem for Linux), It's easy for setup.

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
| Audio Length (min) | Python (FastAPI)  | GO (Fiber) |
| ------------- | ------------- | ------------- |
| 0.5 |  434.46 ms  | 259 ms |
| 1 |  711.39 ms  | 656 ms |
| 5 |  12.474 sec  | 8.169 sec |
| 10 |  26.208 sec  | 15.698 sec |
| 20 |  38.169 sec  | 31.235 sec |
| 30 |  62.582 sec  | 47.805 sec |
| 60 |  122.956 sec  | 102.681 sec |
<image src="/media/end2end.png"/>