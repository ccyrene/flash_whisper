from .onnx.base import ORTModelBase

from .processor.processor import WhisperProcessor

_import_structure = {
    "onnx": [
        "ORTModelBase",
        ],
    }