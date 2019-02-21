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

    def __init__(self, k1, w1, k2, w2, dropout_rate=0.5, do_dropout=True):
        # embedding_dim = D in the paper
        # w1, w2 are the window sizes of the convolutions, hyperparameters
        # ht−1 ∈ R represents information from the previous subtokens m0 . . . mt−1
        super(AttentionFeatures, self).__init__()
        # Use 1D convolutions as input is text.
        self.conv1 = layers.Conv1D(k1, w1, activation='relu')
        self.conv2 = layers.Conv1D(k2, w2)
        self.dropout = layers.Dropout(dropout_rate)
        self.do_dropout = do_dropout

    def call(self, input: List[tf.Tensor], training=False, **kwargs):
        C, h_t = input  # C is code_tokens, h_t is the previous hidden state
        # C = [bodies len, batch size, emb dim]
        # h_t = [1, batch size, k2]
        C = C.permute(1, 2, 0)  # input to conv needs n_channels as dim 1
        h_t = h_t.permute(1, 2, 0)  # from [1, batch size, k2] to [batch size, k2, 1]

        L_1 = self.conv1(C)
        # L_1 = [batch size, k1, bodies len - w1 + 1]
        if self.do_dropout:
            L_1 = self.dropout(L_1, training=training)
        L_2 = self.conv2(L_1) * h_t  # elementwise multiplication
        # L_2 = [batch size, k2, bodies len - w1 - w2 + 2]
        if self.do_dropout:
            L_2 = self.dropout(L_2, training=training)
        # perform L2 normalisation (I suspect that what  L feat <-- L2/||L2||2 means :))
        L_feat = tf.keras.utils.normalize(L_2, order=2)
        return L_feat


class AttentionWeights(tf.keras.Model):
    """
        Accepts L_feat from attention_features and a convolution kernel K of size k2 × w3 ×1.
        Pseudocode from the paper: attention_weights (attention features Lfeat, kernel K):
                return SOFTMAX(CONV1D(Lfeat, K)).
        :returns the normalized attention weights vector with length LEN(c).
    """

    def __init__(self, w3, dropout_rate=0.5, do_dropout=True):
        # TODO experiment with doing dropout here, I don't think it make much sense
        # w3 are the window sizes of the convolutions, hyperparameters
        super(AttentionWeights, self).__init__()
        self.conv1 = layers.Conv1D(1, w3, activation='softmax')
        self.dropout = layers.Dropout(dropout_rate)
        self.do_dropout = do_dropout

    def call(self, l_feat: tf.Tensor, training=False, **kwargs):
        attention_weight = self.conv1(l_feat)
        # L_1 = [batch size, k1, bodies len - w1 + 1]
        if self.do_dropout:
            attention_weight = self.dropout(attention_weight, training=training)
        return attention_weight
