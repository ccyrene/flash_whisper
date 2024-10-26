from .onnx.base import ORTModelBase

from .onnx.processor.processor import WhisperProcessor

_import_structure = {
    "onnx": [
        "ORTModelBase",
        ],
    }