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
                 vocabulary_size,
                 embedding_dim,
                 max_chunk_length,
                 k1,
                 k2,
                 w1,
                 w2,
                 w3,
                 dropout_rate=0.5):
        # TODO experiment with doing dropout here, I don't think it make much sense
        super(ConvAttention, self).__init__()

        self.embedding_layer = layers.Embedding(vocabulary_size,
                                                embedding_dim,
                                                input_length=max_chunk_length)
        self.bias_vector = layers.Embedding(vocabulary_size, 1)

        self.dropout_rate = dropout_rate
        self.masking_layer = layers.Masking(mask_value=0)
        self.gru_layer = layers.GRU(k2, return_state=True)
        self.attention_feature_layer = AttentionFeatures(k1, w1, k2, w2, dropout_rate)
        self.attention_weights_layer = AttentionWeights(w3, dropout_rate)

    def call(self, code_block: tf.Tensor, training=False, **kwargs):
        # input is the body tokens padded and tensorised, and previous state
        tokens = self.embedding_layer(code_block)
        bias = self.bias_vector(code_block)
        _, h_t = self.gru_layer(tokens)  # h_t = [batch_size, units (k2))

        print("ConvAttention: Tokens shape = {}, h_t shape = {}".format(tokens.shape, h_t.shape))
        L_feat = self.attention_feature_layer([tokens, h_t])
        print("ConvAttention: L_feat shape = {}".format(L_feat.shape))

        # L_feat = len(c) + const x k2
        alpha = self.attention_weights_layer(L_feat)
        alpha = self.masking_layer(alpha)  # remove paddings

        n_hat = tf.reduce_sum(alpha * tokens, axis=1)
        # n_hat = [batch size, embedding dimension]
        print("ConvAttention: n_hat shape = {}".format(n_hat.shape))

        softmax = layers.Softmax()(tf.transpose(n_hat * tokens) + bias)
        print("ConvAttention: softmax shape = {}".format(softmax.shape))
        # softmax = [embedding dimension, chunk size, batch size]
        return softmax
