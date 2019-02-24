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
                 dropout_rate=0.5):
        # TODO experiment with doing dropout here, I don't think it make much sense
        super(ConvAttention, self).__init__()

        self.dropout_rate = dropout_rate
        # input already padded from the DLU vocabs library (LookupAndPad is already padded so just lookup)
        # mask padding values, maybe unknown too?
        self.masking_layer = layers.Masking(mask_value=0)
        self.attention_feature_layer = AttentionFeatures(k1, w1, k2, w2, dropout_rate)
        self.attention_weights_layer = AttentionWeights(w3, dropout_rate)

    def call(self, input: List[tf.Tensor], training=False, **kwargs):
        # input is the body tokens padded and tensorised, and previous state
        tokens, h_t = input
        print("ConvAttention: Tokens shape = {}, h_t shape = {}".format(tokens.shape, h_t.shape))
        L_feat = self.attention_feature_layer([tokens, h_t])
        print("ConvAttention: L_feat shape = {}".format(L_feat.shape))

        # L_feat = len(c) + const x k2
        alpha = self.attention_weights_layer(L_feat)
        # print("ConvAttention: alpha shape = {}".format(alpha.shape))
        # alpha_with_emb = tf.keras.backend.transpose(alpha) * tokens
        # print("ConvAttention: alpha_with_emb shape = {}".format(alpha_with_emb.shape))
        # n_hat = tf.reduce_sum(alpha_with_emb)
        #
        # n = layers.Softmax(n_hat)
        # n = self.masking_layer(n)  # remove paddings
        #
        # print("ConvAttention: n shape = {}".format(n.shape))

        return alpha
