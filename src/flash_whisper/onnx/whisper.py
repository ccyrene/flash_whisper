import logging
import numpy as np

from typing import List, Dict, Optional, Union, Tuple
from onnxruntime import InferenceSession

from .base import ORTModelBase
from .utils import ort_type_to_dtype

from ..processor.processor import WhisperProcessor
from ..tokenizer.tokenizer_whisper import WhisperTokenizer

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

        self.key_value_input_names = [key.name for key in self.session.get_inputs() if (".key" in key.name) or (".value" in key.name)]
        self.key_value_output_names = [key.name for key in self.session.get_outputs() if (".key" in key.name) or (".value" in key.name)]
        
    def initial_kv_cache(self, batch_size: int) -> List[np.ndarray]:
        
        inputs = list()
        for model_input in self.session.get_inputs():
            if (".key" in model_input.name) or (".value" in model_input.name) and (".decoder" in model_input.name):
                past_shape = (batch_size, model_input.shape[1], 1, model_input.shape[-1])
                inputs.append(np.zeros(shape=past_shape, dtype=ort_type_to_dtype(model_input.type)))
            elif (".key" in model_input.name) or (".value" in model_input.name) and (".encoder" in model_input.name):
                past_shape = (batch_size, model_input.shape[1], 1, model_input.shape[-1])
                inputs.append(np.zeros(shape=past_shape, dtype=ort_type_to_dtype(model_input.type)))
            
        inputs.append(np.array([False], dtype=bool))

        return inputs
    
    def forward(
        self, 
        input_ids: np.ndarray,
        encoder_hidden_states: np.ndarray,
        use_merged: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[np.ndarray]]] = None,
        ) -> Dict[str, Union[np.ndarray, Tuple[Tuple[np.ndarray]]]]:
        
        batch_size = input_ids.shape[0]
        use_cache = past_key_values is not None
        num_pkv = 4 if use_merged or not use_cache else 2
        
        model_inputs = {
            "input_ids": input_ids,
        }
        
        if use_cache:
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )
        
        if use_merged and use_cache:
            model_inputs["encoder_hidden_states"] = encoder_hidden_states
            model_inputs.update(zip(self.key_value_input_names, past_key_values))
            model_inputs["use_cache_branch"] = np.array([use_cache], dtype=bool)
            
        elif use_merged and not use_cache:
            model_inputs["encoder_hidden_states"] = encoder_hidden_states    
            past_key_values = self.initial_kv_cache(batch_size)
            model_inputs.update(zip(self.key_value_input_names + ["use_cache_branch"], past_key_values))
            
        elif not use_merged and use_cache:
            model_inputs.update(zip(self.key_value_input_names, past_key_values))
            
        else:
            model_inputs["encoder_hidden_states"] = encoder_hidden_states

        out = self.session.run(None, model_inputs)
        out = self.binding_outputs(out)

        out_past_key_values = tuple(out[output_name] for output_name in self.key_value_output_names)

        if not use_cache:
            out_past_key_values = tuple(
                        out_past_key_values[i : i + num_pkv] for i in range(0, len(out_past_key_values), num_pkv)
                    )
        else:
            if num_pkv == 2:
                out_past_key_values = tuple(
                        out_past_key_values[i : i + num_pkv]
                        + past_key_values[2 * i + 2 : 2 * i + 2 + num_pkv]
                        for i in range(0, len(out_past_key_values), num_pkv)
                    )
            else:
                out_past_key_values = tuple(
                            out_past_key_values[i : i + 2] + past_key_values[i + 2 : i + 4]
                            for i in range(0, len(out_past_key_values), num_pkv)
                    )

        return {
            "logits":out["logits"],
            "past_key_values": out_past_key_values,
        }

class ORTWhisper:
    
    processor: WhisperProcessor
    encoder: ORTEncoder
    decoder: ORTDecoder
    tokenizer: WhisperTokenizer

    def __init__(
        self, 
        encoder:InferenceSession, 
        decoder:InferenceSession, 
        decoder_with_past: Optional[InferenceSession] = None
        ):
        
        self.processor = WhisperProcessor()
        self.encoder = ORTEncoder(encoder)
        self.decoder = ORTDecoder(decoder)
        self.tokenizer = WhisperTokenizer()
        
        self.decoder_with_past = ORTDecoder(decoder_with_past) if decoder_with_past is not None else None
        
        self.use_merged = self.decoder_with_past is None
        
    def __call__(self, 
                 audio:np.ndarray, 
                 sampling_rate:int, 
                 language: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        
        input_features = self.processor.extraction(audio, sampling_rate=sampling_rate)["input_features"]
        stopping_criteria = self.processor._get_stopping_criteria()
        input_ids = self.processor._retrieve_init_token(input_features.shape[0])
        
        encoder_hidden_states = self.encoder([input_features])["last_hidden_state"]
        past_key_values = None

        while not np.all(stopping_criteria(input_ids)):
            
            model = self.decoder if self.use_merged or past_key_values is None else self.decoder_with_past
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, encoder_hidden_states, past_key_values)

            decoder_outputs = model(**model_inputs)
            
            next_token = np.argmax(decoder_outputs["logits"][:, -1, :], axis=-1, keepdims=True)
            input_ids = np.concatenate([input_ids, next_token], axis=-1)
            past_key_values = decoder_outputs["past_key_values"]
            
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: np.ndarray,
        encoder_outputs: Optional[np.ndarray] = None,
        past_key_values: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs,
            "use_merged": self.use_merged,
            "past_key_values": past_key_values,
        }