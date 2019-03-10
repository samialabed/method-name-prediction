import logging
from typing import List

from tensorflow import Tensor
from tensorflow.python.keras import backend, layers, models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda, Dropout, Conv1D, Softmax, TimeDistributed


class AttentionFeatures(models.Model):
    """
        <From the paper page 3>
        Attention_features is that given the input c, it uses convolution to compute k2 features for each location.
        By then using ht−1 as a multiplicative gating-like mechanism.
        Only the currently relevant features are kept in L2. In the final stage,

        :arg w1, w2: the window sizes of the convolution
        :arg k1: number of filters on top of the embedding of size w1.
    """

    def __init__(self, k1: int, w1: int, k2: int, w2: int, dropout_rate: float):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        # causal padding to ensure the conv keep the size of the input throughout
        # Keras requires the input to be the same size as the output
        self.conv1 = TimeDistributed(Conv1D(k1, w1, activation='relu', padding='causal', name='attention_fet_conv1'))
        self.conv2 = TimeDistributed(Conv1D(k2, w2, padding='causal', name='attention_fet_conv2'))
        self.dropout = Dropout(dropout_rate)
        self.l2_norm = Lambda(lambda x: backend.l2_normalize(x, axis=1), name='attention_fet_l2_norm')

    def call(self, inputs: List[Tensor], training=False, **kwargs):
        C, h_t = inputs  # C is code_tokens, h_t is the previous hidden state
        self.logger.info("C shape = {}, h_t shape = {}".format(C.shape, h_t.shape))
        # C = [batch size, token length, emb dim]
        # h_t = [batch size, k2], represents information from the previous subtokens m0 . . . mt−1

        L_1 = self.conv1(C)
        self.logger.info("L_1 shape = {}".format(L_1.shape))
        # L_1 = [batch size, token length, k1]
        L_1 = self.dropout(L_1, training=training)
        L_2 = self.conv2(L_1)
        self.logger.info("L_2 shape = {}".format(L_2.shape))
        # elementwise multiplication with h_t to keep only relevant features (acting like a gating-like mechanism)
        L_2 = layers.Multiply(name='attention_fet_l2_mul')([L_2, h_t])

        # L_2 = [batch size, token length, k2]
        self.logger.info("L_2 shape  after multiply = {}".format(L_2.shape))
        L_2 = self.dropout(L_2, training=training)
        # perform L2 normalisation
        L_feat = self.l2_norm(L_2)
        self.logger.info("L_feat shape = {}".format(L_feat.shape))
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
        self.logger = logging.getLogger(__name__)
        self.conv1 = TimeDistributed(Conv1D(1, w3, activation=None, padding='causal', name='atn_weight_conv1'))
        self.dropout = Dropout(dropout_rate)
        self.softmax = TimeDistributed(Softmax(name='atn_weight_softmax'))

    def call(self, l_feat_and_input_mask: List[Tensor], training=False, **kwargs):
        l_feat, mask = l_feat_and_input_mask
        self.logger.info("L_feat shape = {}".format(l_feat.shape))

        attention_weight = self.conv1(l_feat)
        self.logger.info("attention_weight shape = {}".format(attention_weight.shape))
        # attention_weight = [batch size, token length, 1]
        attention_weight = self.dropout(attention_weight, training=training)
        # Give less weights to masked value
        attention_weight = K.squeeze(attention_weight, axis=-1) + mask  # Give less weights to masked value
        attention_weight = self.softmax(attention_weight)
        # attention_weight = [batch size, token length] - what to focus on in the body

        return attention_weight
