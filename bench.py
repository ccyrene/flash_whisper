import optimum

import librosa
import numpy as np

import sys
import os
import time

import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flash_whisper.onnx import ORTWhisper

audio_path = "./sample0.flac"
audio, sr = librosa.load(audio_path)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

whisper_model = ORTWhisper(encoder=ort.InferenceSession("./whisper-tiny/encoder_model.onnx"),
                           decoder=ort.InferenceSession("./whisper-tiny/decoder_model_merged.onnx"))

start_time = time.time()
temp = whisper_model(audio, sampling_rate=16000)
print(f"using time: {(time.time() - start_time) * 1000} ms")

from transformers import AutoProcessor
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("./whisper-tiny")
tokenizer = AutoTokenizer.from_pretrained("./whisper-tiny")
model = ORTModelForSpeechSeq2Seq.from_pretrained("./whisper-tiny")

start_time = time.time()
inputs = processor.feature_extractor(audio, return_tensors="pt")
gen_tokens = model.generate(inputs=inputs.input_features)
temp1 = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
print(f"using time: {(time.time() - start_time) * 1000} ms")


from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

start_time = time.time()
predicted_ids = model.generate(inputs.input_features)
temp2 = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(f"using time: {(time.time() - start_time) * 1000} ms")

print(temp == temp1 == temp2)