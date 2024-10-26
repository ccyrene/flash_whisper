import numpy as np

from typing import Optional, List, Union

class StoppingCriteria:
    def __call__(self, input_ids: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

class StoppingCriteriaList(list):
    def __call__(self, input_ids: np.ndarray, **kwargs) -> np.ndarray:
        is_done = np.full((input_ids.shape[0],), False, dtype=bool)
        for criteria in self:
            is_done = np.logical_or(is_done, criteria(input_ids, **kwargs))
        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None

class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids: np.ndarray, **kwargs) -> np.ndarray:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        return np.full((input_ids.shape[0],), is_done, dtype=bool)

class EosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: Union[int, List[int], np.ndarray]):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: np.ndarray, **kwargs) -> np.ndarray:
        
        is_done = np.isin(input_ids[:, -1], self.eos_token_id)
        
        return is_done