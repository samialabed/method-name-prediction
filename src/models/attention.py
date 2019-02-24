from typing import List

import tensorflow as tf
from tensorflow.python.keras import layers

"""
Attention features and weight as defined in [1] (page 3)


[1] Allamanis, M., Peng, H. and Sutton, C., 2016, June. 
A convolutional attention network for extreme summarization of source code. 
In International Conference on Machine Learning (pp. 2091-2100).
https://arxiv.org/abs/1602.03001
"""


# TODO add loggers instead of print
class AttentionFeatures(tf.keras.Model):
    """
        <From the paper>
        attention_features is that given the input c, it uses convolution to compute k2 features for each location.
        By then using ht−1 as a multiplicative gating-like mechanism, only the
        currently relevant features are kept in L2. In the final stage,
        we normalize L2. attention_features is described with the
        following pseudocode:
        attention_features (code tokens c, context ht−1)
        C ← LOOKUPANDPAD(c, E)
        L1 ← RELU(CONV1D(C, Kl1))
        L2 ← CONV1D(L1, Kl2) * ht−1
        L_feat ← L2/ kL2k2
        return L_feat
    """

    def __init__(self, k1, w1, k2, w2, dropout_rate):
        # embedding_dim = D in the paper
        # w1, w2 are the window sizes of the convolutions, hyperparameters
        # ht−1 ∈ R represents information from the previous subtokens m0 . . . mt−1
        super().__init__()
        # Use 1D convolutions as input is text.
        # create k1 filters, each of window size of w1, the output is k1 different convolutions.
        # causal padding to ensure the conv keep the size of the input throughout
        self.conv1 = layers.Conv1D(k1, w1, activation='relu', padding='causal')
        self.conv2 = layers.Conv1D(k2, w2, padding='causal')
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs: List[tf.Tensor], training=False, **kwargs):
        C, h_t = inputs  # C is code_tokens, h_t is the previous hidden state
        print("AttentionFeatures: C shape = {}, h_t shape = {}".format(C.shape, h_t.shape))
        # C = [batch size, body len (max chunk size), emb dim]
        # h_t = [batch size, k2]

        L_1 = self.conv1(C)
        print("AttentionFeatures: L_1 shape = {}".format(L_1.shape))
        # L_1 = [batch size, bodies len - w1 + 1, k1]
        L_1 = self.dropout(L_1, training=training)
        L_2 = self.conv2(L_1)
        print("AttentionFeatures: L_2 shape = {}".format(L_2.shape))
        # elementwise multiplication with h_t to keep only relevant features (acting like a gating-like mechanism)
        L_2 = L_2 * h_t

        # L_2 = [batch size, k2, bodies len - w1 - w2 + 2]
        print("AttentionFeatures: L_2 shape  after multiply = {}".format(L_2.shape))
        L_2 = self.dropout(L_2, training=training)
        # perform L2 normalisation (I suspect that what  L feat <-- L2/||L2||2 means :))
        L_feat = tf.keras.backend.l2_normalize(L_2)
        print("AttentionFeatures: L_feat shape = {}".format(L_feat))
        return L_feat


class AttentionWeights(tf.keras.Model):
    """
        Accepts L_feat from attention_features and a convolution kernel K of size k2 × w3 ×1.
        Pseudocode from the paper: attention_weights (attention features Lfeat, kernel K):
                return SOFTMAX(CONV1D(Lfeat, K)).
        :returns the normalized attention weights vector with length LEN(c).
    """

    def __init__(self, w3, dropout_rate):
        # w3 are the window sizes of the convolutions, hyperparameters
        super().__init__()
        self.conv1 = layers.Conv1D(1, w3, activation='softmax', padding='causal', use_bias=True)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, l_feat: tf.Tensor, training=False, **kwargs):
        print("AttentionWeights: l_feat shape = {}".format(l_feat.shape))

        attention_weight = self.conv1(l_feat)
        print("AttentionWeights: attention_weight shape = {}".format(attention_weight.shape))
        # L_1 = [batch size, k1, bodies len - w1 + 1]
        attention_weight = self.dropout(attention_weight, training=training)
        return attention_weight
