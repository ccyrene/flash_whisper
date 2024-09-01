import numpy as np

from pathlib import Path
from typing import List, Dict
from abc import abstractmethod
from onnxruntime import InferenceSession

from .utils import ort_type_to_dtype

class ORTModelBase:
    def __init__(self, session:InferenceSession):
        self.session = session
        self.model_path = Path(self.session._model_path)
        self.model_name = self.model_path.name
        
        self.inputs = {idx:{"Name": input_key.name, "Shape": input_key.shape, "Type": input_key.type} for idx, input_key in enumerate(self.session.get_inputs())}
        self.outputs = {idx:{"Name": output_key.name, "Shape": output_key.shape, "Type": output_key.type} for idx, output_key in enumerate(self.session.get_outputs())}
        
    def binding_inputs(self, inputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        session_inputs = self.session.get_inputs()
        if len(inputs) != len(session_inputs):
            raise ValueError("Binding Input Error: Input not compat!")
        
        bindings = {}
        for idx, session_input in enumerate(session_inputs):
            input_idx = inputs[idx]
            
            if ort_type_to_dtype(session_input.type) != input_idx.dtype:
                raise ValueError("Binding Input Error: Type not compat!")

            ort_shape = session_input.shape
            type_mapping = {i: type(element) for i, element in enumerate(ort_shape)}
            for pos, value in type_mapping.items():
                if (value != str) and (ort_shape[pos] != input_idx.shape[pos]):
                    raise ValueError(f"Binding Input Error: Shape not compat!")
            
            bindings[session_input.name] = input_idx
                
        return bindings
    
    def binding_outputs(self, outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        bindings = {}
        for idx, session_output in enumerate(self.session.get_outputs()):
            bindings[session_output.name] = outputs[idx]
        return bindings
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs) 
    
    def __str__(self) -> str:
        return "%s" % self.__class__
    
    def summary(self) -> str:
        """
        showing the (Input&Output)'s Name, Shape and Type of model like table for readable.
        
        example: 
        
            Model Name: encoder_model.onnx
            Model Path: whisper-tiny/encoder_model.onnx

                Inputs:
                                Name       |            Shape            |       Type       |
                        ---------------------------------------------------------------------
                (0)       input_features  |   ['batch_size', 80, 3000]  |  tensor(float)   |
                        ---------------------------------------------------------------------


                Outputs:
                                Name         |            Shape             |       Type       |
                        -------------------------------------------------------------------------
                (0)      last_hidden_state   |  ['batch_size', 1500, 384]   |  tensor(float)   |
                        -------------------------------------------------------------------------
        """
        session_info = f"Model Name: {self.model_name}\n"
        session_info += f"Model Path: {self.model_path}\n"
        session_info += "\n"
        
        air_space = 5
        input_string_length = {
            "Name": max(list(map(len, [str(value["Name"]) for _, value in self.inputs.items()]))) + air_space,
            "Shape": max(list(map(len, [str(value["Shape"]) for _, value in self.inputs.items()]))) + air_space,
            "Type": max(list(map(len, [str(value["Type"]) for _, value in self.inputs.items()]))) + air_space,
        }
        
        output_string_length = {
            "Name": max(list(map(len, [str(value["Name"]) for _, value in self.outputs.items()]))) + air_space,
            "Shape": max(list(map(len, [str(value["Shape"]) for _, value in self.outputs.items()]))) + air_space,
            "Type": max(list(map(len, [str(value["Type"]) for _, value in self.outputs.items()]))) + air_space,
        }

        first_indent = 10
        n_character_input_line = input_string_length["Name"] + input_string_length["Shape"] + input_string_length["Type"] + 3
        n_character_output_line = output_string_length["Name"] + output_string_length["Shape"] + output_string_length["Type"] + 3
        
        session_info += "Inputs:\n"
        session_info += " " * first_indent
        session_info += "Name".center(input_string_length["Name"]) + "|"
        session_info += "Shape".center(input_string_length["Shape"]) + "|"
        session_info += "Type".center(input_string_length["Type"]) + "|\n"
        session_info += " " * first_indent
        session_info += "-" * n_character_input_line
        session_info += "\n"
        
        for key, value in self.inputs.items():
            session_info += f"({key})".center(first_indent)
            session_info += str(value['Name']).center(input_string_length['Name']) + "|"
            session_info += str(value['Shape']).center(input_string_length['Shape']) + "|"
            session_info += str(value['Type']).center(input_string_length['Type']) + "|\n"

        session_info += " " * first_indent
        session_info += "-" * n_character_input_line
        session_info += "\n" * 3
        
        session_info += "Outputs:\n"
        session_info += " " * first_indent
        session_info += "Name".center(output_string_length["Name"]) + "|"
        session_info += "Shape".center(output_string_length["Shape"]) + "|"
        session_info += "Type".center(output_string_length["Type"]) + "|\n"
        session_info += " " * first_indent
        session_info += "-" * n_character_output_line
        session_info += "\n"
        
        for key, value in self.outputs.items():
            session_info += f"({key})".center(first_indent)
            session_info += str(value['Name']).center(output_string_length['Name']) + "|"
            session_info += str(value['Shape']).center(output_string_length['Shape']) + "|"
            session_info += str(value['Type']).center(output_string_length['Type']) + "|\n"

        session_info += " " * first_indent
        session_info += "-" * n_character_output_line
        session_info += "\n"
        
        return session_info
    