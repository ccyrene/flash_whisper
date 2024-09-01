import logging
import numpy as np

from typing import List, Dict, Optional, Union
from onnxruntime import InferenceSession

from .base import ORTModelBase
from .utils import ort_type_to_dtype

from ..processor.processor import WhisperProcessor

logger = logging.getLogger(__name__)

class ORTEncoder(ORTModelBase):
    
    def __init__(self, session: InferenceSession):
        super().__init__(session)
    
    def forward(self, inputs: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        out = self.session.run(None, self.binding_inputs(inputs))
        return self.binding_outputs(out)

class ORTDecoder(ORTModelBase):
    def __init__(self, session: InferenceSession):
        super().__init__(session)
        
    def initial_kv_cache(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        
        batch_size, past_decoder_sequence_length = inputs[0].shape
        encoder_sequence_length_out = inputs[1].shape[1]

        for model_input in self.session.get_inputs()[len(inputs):-1]:
            if "decoder" in model_input.name:
                past_shape = (batch_size, model_input.shape[1], past_decoder_sequence_length, model_input.shape[-1])
            else:
                past_shape = (batch_size, model_input.shape[1], encoder_sequence_length_out, model_input.shape[-1])

            inputs.append(np.zeros(shape=past_shape, dtype=ort_type_to_dtype(model_input.type)))
            
        inputs.append(np.array([False], dtype=bool))
        
        return inputs
    
    def forward(self, inputs: List[np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        use_cache = len(self.session.get_inputs()) == len(inputs)
        
        if not use_cache:
            inputs = self.initial_kv_cache(inputs)
            
        inputs = self.binding_inputs(inputs)
        
        print("======INPUT==========")
        for key, value in inputs.items():
            print(f"{key}: {value.shape}")
            
        out = self.session.run(None, inputs)
        out = self.binding_outputs(out)
        
        print("======OUTPUT==========")
        for key, value in out.items():
            print(f"{key}: {value.shape}")
        
        return out

class ORTWhisper:
    
    processor: WhisperProcessor
    encoder: ORTEncoder
    decoder: ORTDecoder
    
    def __init__(self, encoder:InferenceSession, decoder:InferenceSession):
        self.processor = WhisperProcessor()
        self.encoder = ORTEncoder(encoder)
        self.decoder = ORTDecoder(decoder)
    
    def __call__(self, 
                 audio:np.ndarray, 
                 sampling_rate:int, 
                 language: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        
        input_features = self.processor.extraction(audio, sampling_rate=sampling_rate)["input_features"]
        stopping_criteria = self.processor._get_stopping_criteria()
        decoder_input_ids = self.processor._retrieve_init_token(input_features.shape[0])
        
        encoder_hidden_states = self.encoder([input_features])["last_hidden_state"]
        
        temp = self.decoder([decoder_input_ids, encoder_hidden_states])
        
        return temp