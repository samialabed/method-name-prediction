from tensorflow.python import keras, Tensor

from models.attention import AttentionFeatures, AttentionWeights


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
                                                      input_length=max_chunk_length)
        self.masking_layer = keras.layers.Masking(mask_value=0)

        self.gru_layer = keras.layers.GRU(k2, return_state=True, stateful=True)
        self.attention_feature_layer = AttentionFeatures(k1, w1, k2, w2, dropout_rate)
        self.attention_weights_layer = AttentionWeights(w3, dropout_rate)
        # dense layer: n ← vector sum of Ei * ai. Followed by E * n_t + bias, mapped to probability of words embedding
        self.dense_layer = keras.layers.Dense(vocabulary_size, activation='softmax')

    def call(self, code_block: Tensor, training=False, **kwargs):
        # input is the body tokens_embedding padded and tensorised, and previous state
        code_block = self.masking_layer(code_block)
        tokens_embedding = self.embedding_layer(code_block)
        # if testing: tf.random > drop_rate: self.gru_layer(predicted embedding)
        _, h_t = self.gru_layer(tokens_embedding, training=training)  # h_t = [batch_size, units (k2))

        print("ConvAttention: Tokens shape = {}, h_t shape = {}".format(tokens_embedding.shape, h_t.shape))
        l_feat = self.attention_feature_layer([tokens_embedding, h_t])
        print("ConvAttention: L_feat shape = {}".format(l_feat.shape))

        # L_feat = len(c) + const x k2
        alpha = self.attention_weights_layer(l_feat)
        n = self.dense_layer(alpha)
        print("ConvAttention: n shape = {}".format(n.shape))
        # n = [embedding dimension, chunk size, batch size]
        return n
