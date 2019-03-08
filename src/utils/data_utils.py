from typing import List

import numpy as np
from dpu_utils.mlutils import Vocabulary
from tensorflow.python.keras import backend as K


def translate_tokenized_array_to_list_words(vocab: Vocabulary, token: np.ndarray) -> List[str]:
    """Helper function to translate numpy array tokens back to words"""
    return [vocab.get_name_for_id(n) for n in filter(lambda f: f != 0, token)]


def clean_target_from_padding(target: np.ndarray):
    """Helper function to remove the padding and put the target array in easy to use format"""
    return [np.trim_zeros(x.flatten(), 'b') for x in target]


def beam_search(predictions: np.ndarray, y: np.ndarray,
                padding_token_id: int, start_sentence_token_id: int, end_sentence_token_id: int,
                beam_width: int = 5, beam_top_paths: int = 5):
    """
    predictions: output from a softmax layer, y true labels
    # TODO if time permits implement own beam search, TF is too slow
    """
    best_predictions = []
    best_predictions_probs = []
    for p in predictions:
        val, probs = K.ctc_decode(
            np.expand_dims(p, 0),
            (y.shape[1],),
            greedy=False,
            beam_width=beam_width,
            top_paths=beam_top_paths
        )
        # evaluate tensorflow graph

        x_pred = K.batch_get_value(val)
        # remove paddings and ensure start and end token exist
        top_paths_predictions = [trim_pred(pred,
                                           padding_token_id,
                                           start_sentence_token_id,
                                           end_sentence_token_id) for pred in x_pred]

        best_predictions.append(top_paths_predictions)
        top_paths_predictions_probs = K.get_value(probs)
        best_predictions_probs.append(np.exp(top_paths_predictions_probs[0]))
    return best_predictions, best_predictions_probs


def trim_pred(pred: np.ndarray, padding_id: int, start_sentence_token_id: int, end_sentence_token_id: int):
    """Ensures start and end token in prediction, trim zeros"""
    padding_removed = pred[np.nonzero(pred != padding_id)]

    if padding_removed[0] != start_sentence_token_id:
        padding_removed = np.insert(padding_removed, 0, start_sentence_token_id)
    for idx, p in enumerate(padding_removed):
        if p == end_sentence_token_id:
            return padding_removed[: idx + 1]  # stop at sentence end
    # no sentence end detected, add it manually
    return np.append(padding_removed, end_sentence_token_id)
