from timeit import timeit

setup = """import torch

from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

model_path = "/workspace/whisper-medium.en"
device = "cuda"

processor = WhisperProcessor.from_pretrained(
          model_path, language='th', 
          task="transcribe", 
          predict_timestamps=False)

model = WhisperForConditionalGeneration.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                    use_safetensors=True)

model.eval()
model.to(device)

hug_pipe = pipeline("automatic-speech-recognition", 
                    model=model, 
                    tokenizer=processor.tokenizer, 
                    feature_extractor=processor.feature_extractor, 
                    max_new_tokens=128, 
                    chunk_length_s=30,
                    batch_size=4, 
                    return_timestamps=False,
                    torch_dtype=torch.float16, 
                    device=device)
"""

method = """hug_pipe("/workspace/sample0.flac")['text']"""

n = 10 
print(timeit(setup=setup, stmt=method, number=n) / n)