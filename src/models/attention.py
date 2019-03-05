from typing import List

from tensorflow import Tensor
from tensorflow.python.keras import backend, layers, models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda, Dropout, Conv1D, Softmax, TimeDistributed

"""
Attention features and weight as defined in [1] (page 3)


[1] Allamanis, M., Peng, H. and Sutton, C., 2016, June. 
A convolutional attention network for extreme summarization of source code. 
In International Conference on Machine Learning (pp. 2091-2100).
https://arxiv.org/abs/1602.03001
"""


# TODO add loggers instead of print
class AttentionFeatures(models.Model):
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

    def __init__(self, k1: int, w1: int, k2: int, w2: int, dropout_rate: float):
        # embedding_dim = D in the paper
        # w1, w2 are the window sizes of the convolutions, hyperparameters
        # ht−1 ∈ R represents information from the previous subtokens m0 . . . mt−1
        super().__init__()
        # Use 1D convolutions as input is text.
        # create k1 filters, each of window size of w1, the output is k1 different convolutions.
        # causal padding to ensure the conv keep the size of the input throughout
        # Keras requires the input to be the same size as the output
        self.conv1 = TimeDistributed(Conv1D(k1, w1, activation='relu', padding='causal', name='attention_fet_conv1'))
        self.conv2 = TimeDistributed(Conv1D(k2, w2, padding='causal', name='attention_fet_conv2'))
        self.dropout = Dropout(dropout_rate)
        self.l2_norm = Lambda(lambda x: backend.l2_normalize(x, axis=1), name='attention_fet_l2_norm')

    def call(self, inputs: List[Tensor], training=False, **kwargs):
        C, h_t = inputs  # C is code_tokens, h_t is the previous hidden state
        print("AttentionFeatures: C shape = {}, h_t shape = {}".format(C.shape, h_t.shape))
        # C = [batch size, max chunk size, emb dim]
        # h_t = [batch size, k2]

        L_1 = self.conv1(C)
        print("AttentionFeatures: L_1 shape = {}".format(L_1.shape))
        # L_1 = [batch size, max chunk size, k1]
        L_1 = self.dropout(L_1, training=training)
        L_2 = self.conv2(L_1)
        print("AttentionFeatures: L_2 shape = {}".format(L_2.shape))
        # elementwise multiplication with h_t to keep only relevant features (acting like a gating-like mechanism)
        L_2 = layers.Multiply(name='attention_fet_l2_mul')([L_2, h_t])

        # L_2 = [batch size, max chunk size, k2]
        print("AttentionFeatures: L_2 shape  after multiply = {}".format(L_2.shape))
        L_2 = self.dropout(L_2, training=training)
        # perform L2 normalisation
        L_feat = self.l2_norm(L_2)
        print("AttentionFeatures: L_feat shape = {}".format(L_feat.shape))
        return L_feat


class AttentionWeights(models.Model):
    """
        Accepts L_feat from attention_features and a convolution kernel K of size k2 × w3 ×1.
        Pseudocode from the paper: attention_weights (attention features Lfeat, kernel K):
                return SOFTMAX(CONV1D(Lfeat, K)).
        :returns the normalized attention weights vector with length LEN(c).
    """

    def __init__(self, w3, dropout_rate):
        # w3 are the window sizes of the convolutions, hyperparameters
        super().__init__()
        self.conv1 = TimeDistributed(
            Conv1D(1, w3, activation=None, padding='causal', use_bias=True, name='atn_weight_conv1'))
        self.dropout = Dropout(dropout_rate)
        self.softmax = Softmax(name='atn_weight_softmax')

    def call(self, l_feat_and_input_mask: List[Tensor], training=False, **kwargs):
        l_feat, mask = l_feat_and_input_mask
        print("AttentionWeights: l_feat shape = {}".format(l_feat.shape))

        attention_weight = self.conv1(l_feat)
        print("AttentionWeights: attention_weight shape = {}".format(attention_weight.shape))
        # attention_weight = [batch size, max chunk size, 1]
        attention_weight = self.dropout(attention_weight, training=training)
        # Give less weights to masked value
        attention_weight = K.squeeze(attention_weight, axis=-1) + mask  # Give less weights to masked value
        attention_weight = self.softmax(attention_weight)
        # attention_weight = [batch size, max chunk size] - what to focus on in the body

        return attention_weight
