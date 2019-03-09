import os
import time
from typing import List

import numpy as np
from dpu_utils.mlutils import Vocabulary
from tensorflow.python.keras import backend as K


# TODO consider moving Beam related utils to a beam object

def translate_tokenized_array_to_list_words(vocab: Vocabulary, token: np.ndarray) -> List[str]:
    """Helper function to translate numpy array tokens back to words"""
    return [vocab.get_name_for_id(n) for n in token[np.nonzero(token != vocab.get_id_or_unk(vocab.get_pad()))]]


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
    print("{}: In beam search".format(time.strftime(time.strftime("%Y-%m-%d-%H-%M%S"))))
    start_time = time.time()

    beam_search_predictions_list = []
    beam_search_probs_list = []
    for pred in predictions:
        top_path_prediction_tensors, probs = K.ctc_decode(
            np.expand_dims(pred, 0),
            (y.shape[1],),
            greedy=False,
            beam_width=beam_width,
            top_paths=beam_top_paths
        )
        beam_search_predictions_list.append(top_path_prediction_tensors)
        beam_search_probs_list.append(probs)

    # evaluate tensorflow graph
    print("{}: Evaluating beam search TF graph".format(time.strftime(time.strftime("%Y-%m-%d-%H-%M%S"))))
    beam_search_predictions_evaluated: List[np.ndarray] = K.batch_get_value(beam_search_predictions_list)
    print("{} Cleaning beamsearch results".format(time.strftime(time.strftime("%Y-%m-%d-%H-%M%S"))))
    best_predictions = [list(trim_pred(pred, padding_token_id,
                                       start_sentence_token_id,
                                       end_sentence_token_id) for pred in beam_search_single_result)
                        for beam_search_single_result in beam_search_predictions_evaluated]
    del beam_search_predictions_evaluated  # freeup much needed memory
    top_paths_predictions: np.ndarray = K.batch_get_value(beam_search_probs_list)
    best_predictions_probs = list(map(lambda pred: np.exp(pred[0]), top_paths_predictions))
    del top_paths_predictions  # freeup much needed memory
    print("beam search ended for one iteration in {}ms".format(time.time() - start_time))
    return best_predictions, best_predictions_probs


def trim_pred(pred: np.ndarray,
              padding_id: int,
              start_sentence_token_id: int,
              end_sentence_token_id: int) -> np.ndarray:
    """Ensures start and end token in prediction, trim zeros"""
    padding_removed = pred[np.nonzero(pred != padding_id)]
    if padding_removed.shape[0] == 0:
        pred[0] = 1
        return pred[0][:1]

    if padding_removed[0] != start_sentence_token_id:
        padding_removed = np.insert(padding_removed, 0, start_sentence_token_id)
    for idx, p in enumerate(padding_removed):
        if p == end_sentence_token_id:
            return padding_removed[: idx + 1]  # stop at sentence end
        if p == -1:
            padding_removed[idx] = 1  # map the ctc_decode -1 'unknown' representation to the vocab's one
    # no sentence end detected, add it manually

    return np.append(padding_removed, end_sentence_token_id)


def visualise_beam_predictions_to_targets(vocab,
                                          best_predictions: List[np.ndarray],
                                          best_predictions_probs: List[np.ndarray],
                                          input_method_body_subtokens: np.ndarray,
                                          target_method_names: np.ndarray):
    target_methods_translated = [translate_tokenized_array_to_list_words(vocab, target_method_name) for
                                 target_method_name in target_method_names]

    input_body_translated = [translate_tokenized_array_to_list_words(vocab, input_method_body_subtoken) for
                             input_method_body_subtoken in input_method_body_subtokens]

    best_predictions_translated = [
        list(translate_tokenized_array_to_list_words(vocab, pred) for pred in best_prediction)
        for best_prediction in best_predictions]

    results = []
    for input_body, target_name, predictions, probs in zip(input_body_translated, target_methods_translated,
                                                           best_predictions_translated, best_predictions_probs):
        results.append('==================Begin Words==============================={}'.format(os.linesep))
        results.append('input_body: {}{}'.format(input_body, os.linesep))
        results.append('target_name: {}{}'.format(target_name, os.linesep))
        results.append('predictions: {}{}'.format(predictions, os.linesep))
        results.append('probs: {}{}'.format(probs, os.linesep))
        results.append('================================================={}'.format(os.linesep))

    return ''.join(results)
