from typing import List

import tensorflow as tf
from tensorflow.python.keras import layers

from models.attention import AttentionFeatures, AttentionWeights


class ConvAttention(tf.keras.Model):
    """
    <From the paper>
    conv_attention, a convolutional attentional model that uses
    an attention vector α computed from attention_weights to
    weight the embeddings of the tokens in c and compute the
    predicted target embedding ˆn ∈ R
    D. It returns a distribution
    over all subtokens in V .
    """

    def __init__(self,
                 k1,
                 k2,
                 w1,
                 w2,
                 w3,
                 dropout_rate=0.5,
                 do_dropout=True):
        # TODO experiment with doing dropout here, I don't think it make much sense
        super(ConvAttention, self).__init__()

        self.do_dropout = do_dropout
        self.dropout_rate = dropout_rate

        # input already padded from the DLU vocabs library (LookupAndPad is already padded so just lookup)
        # mask padding values, maybe unknown too?
        self.masking_layer = layers.Masking(mask_value=0)
        self.attention_feature_layer = AttentionFeatures(k1, w1, k2, w2, dropout_rate, do_dropout)
        self.attention_weights_layer = AttentionWeights(w3, dropout_rate, do_dropout)

    def call(self, input: List[tf.Tensor], training=False, **kwargs):
        # input is the body tokens padded and tensorised, and previous state
        tokens, h_t = input
        L_feat = self.attention_feature_layer([tokens, h_t])
        # L_feat = len(c) + const x k2
        alpha = self.attention_weights_layer(L_feat)
        n_hat = tf.reduce_sum(alpha * tokens)  # this doesn't look right?
        n = layers.Softmax(n_hat.T)
        n = self.masking_layer(n)  # remove paddings
        return n
