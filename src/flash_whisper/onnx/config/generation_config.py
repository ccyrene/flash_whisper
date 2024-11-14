import os
import json

class GenerationConfig:
    def __init__(self, config_dir, **kwargs):
        with open(os.path.join(config_dir, 'generation_config.json'), "r") as f:
            config = json.load(f)

        for key, value in config.items():
            setattr(self, key, value)
        self.use_cache = kwargs.pop("use_cache", True)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)

    def update(self, **kwargs):
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs
    