import tensorflow as tf
from tensorflow.python import keras, Tensor

from models.attention import AttentionFeatures, AttentionWeights
from utils.activations import LogSoftmax


class ConvAttention(keras.Model):
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
        super().__init__()
        self.dropout_rate = dropout_rate

        self.embedding_layer = keras.layers.Embedding(vocabulary_size,
                                                      embedding_dim,
                                                      input_length=max_chunk_length,
                                                      name='cnn_att_embedding')

        self.gru_layer = keras.layers.GRU(k2, return_state=True, stateful=True, name='cnn_att_gru')
        self.attention_feature_layer = AttentionFeatures(k1, w1, k2, w2, dropout_rate)
        self.attention_weights_layer = AttentionWeights(w3, dropout_rate)
        # dense layer: E * n_t + bias, mapped to probability of words embedding
        self.dense_layer = keras.layers.Dense(vocabulary_size, activation='softmax', name='cnn_att_dense')
        # self.dense_layer = keras.layers.Dense(vocabulary_size, activation=LogSoftmax(), name='cnn_att_dense')

    def call(self, code_block: Tensor, training=False, **kwargs):
        # input is the body tokens_embedding padded and tensorised, and previous state
        mask_vector = tf.cast(tf.equal(code_block, 0), dtype=tf.float32)  # Shape [batch size, chunk length]
        tokens_embedding = self.embedding_layer(code_block)
        # if testing: tf.random > drop_rate: self.gru_layer(predicted embedding)
        _, h_t = self.gru_layer(tokens_embedding, training=training)  # h_t = [batch_size, units (k2))

        print("ConvAttention: Tokens shape = {}, h_t shape = {}".format(tokens_embedding.shape, h_t.shape))
        l_feat = self.attention_feature_layer([tokens_embedding, h_t])
        print("ConvAttention: L_feat shape = {}".format(l_feat.shape))

        # L_feat = len(c) + const x k2
        alpha = self.attention_weights_layer([l_feat, mask_vector])
        print("ConvAttention: alpha shape = {}".format(alpha.shape))
        # alpha = [batch size, embedding] weights over embeddings
        n_hat = tf.reduce_sum(keras.layers.Multiply()([tf.expand_dims(alpha, axis=-1), tokens_embedding]), axis=-1)
        print("ConvAttention: n_hat shape = {}".format(n_hat.shape))
        # n_hat = [batch size, embedding]

        n = self.dense_layer(tokens_embedding * tf.transpose(n_hat))
        print("ConvAttention: n shape = {}".format(n.shape))
        # n = [batch size, embedding, vocabulary size] probability of each token appearing in the embedding

        # n = tf.reduce_mean(n, axis=2)
        # print("ConvAttention: n shape = {}".format(n.shape))

        return n
