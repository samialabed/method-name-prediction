from typing import List

import numpy as np
from dpu_utils.mlutils import Vocabulary


def translate_tokenized_array_to_list_words(vocab: Vocabulary, token: np.ndarray) -> List[str]:
    return [vocab.get_name_for_id(n) for n in filter(lambda f: f != 0, token)]
