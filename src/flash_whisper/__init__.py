from .onnx.base import ORTModelBase

from .onnx.processor.processor import WhisperProcessor
from .tllm.whisper import WhisperTRTLLM

_import_structure = {
    "onnx": [
        "ORTModelBase",
        ],
    
    "tllm": [
        "WhisperTRTLLM"
        ],
    }