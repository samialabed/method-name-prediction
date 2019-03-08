import os
from typing import List, Dict

import numpy as np
from dpu_utils.mlutils import Vocabulary
from scipy.integrate import simps
from tensorflow.python import keras

from data.constants import SENTENCE_END_TOKEN, SENTENCE_START_TOKEN
from utils.data_utils import beam_search, clean_target_from_padding


def evaluate_f1(model: keras.Model, vocab: Vocabulary, x: np.ndarray, y: np.ndarray, hyperparameter: Dict[str, any]):
    if x.ndim != 3:
        # model prediction expects 3 dimensions, a single input won't have the batch dimension, manually add it
        x = np.expand_dims(x, 0)

    predictions = model.predict(x)

    padding_id = vocab.get_id_or_unk(vocab.get_pad())
    begin_of_sentence_id = vocab.get_id_or_unk(SENTENCE_START_TOKEN)
    end_of_sentence_id = vocab.get_id_or_unk(SENTENCE_END_TOKEN)

    best_predictions, best_predictions_probs = beam_search(predictions, y,
                                                           padding_id,
                                                           begin_of_sentence_id,
                                                           end_of_sentence_id,
                                                           hyperparameter['beam_width'],
                                                           hyperparameter['beam_top_paths'],
                                                           )
    # return best_predictions, best_predictions_probs
    return _evaluate_f1(best_predictions, best_predictions_probs, vocab, y)


def _evaluate_f1(best_predictions: List[List[np.ndarray]],
                 best_predictions_probs: List[np.ndarray],
                 vocab: Vocabulary,
                 true_labels: np.ndarray):
    true_labels = clean_target_from_padding(true_labels)
    result_accumulator = PointSuggestionEvaluator()
    unk_id = vocab.get_id_or_unk(vocab.get_unk())

    for x_pred, x_prob, y_target in zip(best_predictions, best_predictions_probs, true_labels):
        confidences = x_prob.tolist()
        is_exact_prediction = [np.all(pred == y_target) for pred in x_pred]
        precision_recall = [token_precision_recall(pred.T, y_target) for pred in x_pred]
        is_unknown_word_predicted = [np.all(suggestion == unk_id) for suggestion in x_pred]
        unk_word_accuracy = [unk_acc(suggestion.T, y_target, unk_id) for suggestion in x_pred]
        result_accumulator.add_result(confidences, is_exact_prediction, is_unknown_word_predicted, precision_recall,
                                      unk_word_accuracy)

    return result_accumulator


def unk_acc(suggested_subtokens, real_subtokens, unk_id):
    real_unk_subtokens = np.sum(real_subtokens == unk_id)
    if real_unk_subtokens == 0:
        return None
    return float(np.sum(suggested_subtokens == unk_id)) / real_unk_subtokens


# TODO this is super hacky and probably inefficient
class PointSuggestionEvaluator:
    """
    This a modified version from f1_evaluator from
    https://github.com/mast-group/convolutional-attention/blob/master/convolutional_attention/f1_evaluator.py
    """

    def __init__(self):
        self.confidence_threshold = [0, 0.001, 0.005, 0.01, 0.02, 0.04, 0.05]
        self.rank_to_eval = [1, 5]
        self.num_points = 0
        self.num_made_suggestions = np.array([[0] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.num_correct_suggestions = np.array([[0] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_precisions_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_recalls_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_f1_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_unk_word_accuracy = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_unk_word_locations = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))

    def get_f1_at_all_ranks(self):
        """
        Get the F1 score, when all tokens are suggested at the self.rank_to_eval ranks
        :rtype: list
        :return: a list of the f1 scores
        """
        return self.sum_f1_suggestions[:, 0] / self.num_points

    def add_result(self, confidence, is_correct, is_unk, precision_recall, unk_word_accuracy):
        """
        Add a single point suggestion as a result.
        """
        confidence = np.array(confidence)
        is_correct = np.array(is_correct, dtype=np.bool)
        is_unk = np.array(is_unk, dtype=np.bool)
        self.num_points += 1
        if len(is_unk) == 0 or is_unk[0]:
            return  # No suggestions
        for i in range(len(self.confidence_threshold)):
            # How many probabilities are above the threshold (probs are sorted desc)
            num_confident_suggestions = confidence[confidence >= self.confidence_threshold[i]].shape[0]
            for j in range(len(self.rank_to_eval)):
                rank = self.rank_to_eval[j]
                n_suggestions = min(rank, num_confident_suggestions)

                unk_at_rank = np.where(is_unk[:n_suggestions])[0]
                if unk_at_rank.shape[0] == 0:
                    unk_at_rank = n_suggestions + 1  # Beyond our current number of suggestions
                else:
                    unk_at_rank = unk_at_rank[0]

                if min(n_suggestions, unk_at_rank) > 0:
                    self.num_made_suggestions[j][i] += 1
                    if np.any(is_correct[:min(n_suggestions, unk_at_rank)]):
                        self.num_correct_suggestions[j][i] += 1

                    pr, re, f1 = self.get_best_f1(precision_recall[:min(n_suggestions, unk_at_rank)])
                    self.sum_precisions_suggestions[j][i] += pr
                    self.sum_recalls_suggestions[j][i] += re
                    self.sum_f1_suggestions[j][i] += f1

                unk_accuracies = [s for s in unk_word_accuracy[:min(n_suggestions, unk_at_rank)] if s is not None]
                if len(unk_accuracies) > 0:
                    # There is at least one UNK here
                    self.sum_unk_word_locations[j][i] += 1
                    self.sum_unk_word_accuracy[j][i] += max(unk_accuracies)

    def get_best_f1(self, suggestions_pr_re_f1):
        """
        Get the "best" precision, recall and f1 score from a list of tuples,
        picking the ones with the best f1
        """
        max_f1 = 0
        max_pr = 0
        max_re = 0
        for suggestion in suggestions_pr_re_f1:
            if suggestion[2] > max_f1:
                max_pr, max_re, max_f1 = suggestion
        return max_pr, max_re, max_f1

    def __str__(self):
        n_made_suggestions = np.array(self.num_made_suggestions, dtype=float)
        n_correct_suggestions = np.array(self.num_correct_suggestions, dtype=float)
        results_list = []
        for i in range(len(self.rank_to_eval)):
            rank_str = 'At Rank {}{}'.format(self.rank_to_eval[i], os.linesep)
            sug_freq = 'Suggestion Frequency {}{}'.format((n_made_suggestions[i] / self.num_points), os.linesep)
            sug_acc = 'Suggestion Accuracy {}{}'.format(np.divide(n_correct_suggestions[i], n_made_suggestions[i]),
                                                        os.linesep)
            unk_acc = 'UNK Accuracy {}{}'.format(
                np.divide(self.sum_unk_word_accuracy[i], self.sum_unk_word_locations[i]), os.linesep)

            sug_prec = 'Suggestion Precision {}{}'.format(
                np.divide(self.sum_precisions_suggestions[i], n_made_suggestions[i]), os.linesep)
            sug_recall = 'Suggestion Recall {}{}'.format(
                np.divide(self.sum_recalls_suggestions[i], n_made_suggestions[i]), os.linesep)
            sug_f1 = 'Suggestion F1 {}{}'.format(np.divide(self.sum_f1_suggestions[i], n_made_suggestions[i]),
                                                 os.linesep)
            num_points = 'Num Points: {}{}'.format(self.num_points, os.linesep)
            results_list.append(rank_str)
            results_list.append(sug_freq)
            results_list.append(sug_acc)
            results_list.append(unk_acc)
            results_list.append(sug_prec)
            results_list.append(sug_recall)
            results_list.append(sug_f1)
            results_list.append(num_points)

        return ''.join(results_list)

    def get_f1_auc(self, rank_idx=0):
        n_made_suggestions = np.array(self.num_made_suggestions, dtype=float)
        f1_at_rank = np.divide(self.sum_f1_suggestions[rank_idx], n_made_suggestions[rank_idx])
        suggestion_freq = n_made_suggestions[rank_idx] / self.num_points

        mask = np.bitwise_not(np.isnan(f1_at_rank))
        unique_freq, unique_idx = np.unique(suggestion_freq[mask][::-1], return_index=True)
        unique_freq = unique_freq[::-1]
        f1_at_rank = f1_at_rank[mask][::-1][unique_idx][::-1]

        if len(unique_freq) > 0:
            return -simps(f1_at_rank, unique_freq)
        return 0

    def get_acc_auc(self, rank_idx=0):
        n_made_suggestions = np.array(self.num_made_suggestions, dtype=float)
        acc_at_rank = np.divide(self.num_correct_suggestions[rank_idx], n_made_suggestions[rank_idx])
        suggestion_freq = n_made_suggestions[rank_idx] / self.num_points
        mask = np.bitwise_not(np.isnan(acc_at_rank))
        unique_freq, unique_idx = np.unique(suggestion_freq[mask][::-1], return_index=True)
        unique_freq = unique_freq[::-1]

        acc_at_rank = acc_at_rank[mask][::-1][unique_idx][::-1]
        if len(unique_freq) > 0:
            return -simps(acc_at_rank, unique_freq)
        return 0


def token_precision_recall(predicted_parts: np.ndarray, gold_set_parts: np.ndarray):
    """
    Get the precision/recall for the given token.
    :param predicted_parts: a list of predicted parts
    :param gold_set_parts: a list of the golden parts
    :return: precision, recall, f1 as floats
    """

    tp = len(np.intersect1d(predicted_parts, gold_set_parts))
    assert tp <= len(predicted_parts), (tp, len(predicted_parts), predicted_parts, gold_set_parts)
    if len(predicted_parts) > 0:
        precision = float(tp) / len(predicted_parts)
    else:
        precision = 0

    assert tp <= len(gold_set_parts), (tp, gold_set_parts, predicted_parts)
    if len(gold_set_parts) > 0:
        recall = float(tp) / len(gold_set_parts)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.

    return precision, recall, f1


def _debug_evaluator():
    from collections import Counter

    test_gold = np.array(
        [[[10], [130], [97], [377], [74], [11], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
          [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
          [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]])

    test_prediction = [np.array([[[10, 130, 377, 1, 0]],
                                 [[10, 130, 377, 0]],
                                 [[10, 12, 377, 1, 0]],
                                 [[10, 130, 377, 74, 0]],
                                 [[10, 12, 377, 0]]]),
                       np.array([[[10, 130, 377, 11, 0]],
                                 [[10, 130, 377, 0]],
                                 [[10, 12, 377, 11, 0]],
                                 [[10, 130, 377, 74, 0]],
                                 [[10, 12, 377, 0]]])]

    # token_precision_recall(test_prediction, test_gold, -1)

    test_prediction_prob = [np.array([-5.6808667, -5.8504624, -5.9559455, -6.125316, -6.1794915],
                                     dtype='float32'),
                            np.array([-3.8716054, -3.9793577, -4.3363566, -6.695883, -8.953899], dtype='float32')]

    test_prediction_prob = np.exp(test_prediction_prob)

    test_vocab = Vocabulary.create_vocabulary(Counter([10, 130, 377, 11, 0]),
                                              count_threshold=0,
                                              max_size=50,
                                              add_unk=True,
                                              add_pad=True)

    print(_evaluate_f1(test_prediction, test_prediction_prob, test_vocab, test_gold))


if __name__ == '__main__':
    _debug_evaluator()
