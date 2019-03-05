from typing import Dict

from tensorflow.python import keras, Tensor
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, GRU, TimeDistributed, Softmax

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

    def __init__(self, hyperparameters: Dict[str, any]):
        super().__init__()
        vocabulary_size = hyperparameters['vocabulary_size']
        embedding_dim = hyperparameters['embedding_dim']
        max_chunk_length = hyperparameters['max_chunk_length']
        dropout_rate = hyperparameters['dropout_rate']
        w1 = hyperparameters['w1']
        w2 = hyperparameters['w2']
        w3 = hyperparameters['w3']
        k1 = hyperparameters['k1']
        k2 = hyperparameters['k2']
        self.embedding_layer = TimeDistributed(Embedding(vocabulary_size,
                                                         embedding_dim,
                                                         mask_zero=True,
                                                         input_length=max_chunk_length,
                                                         name='cnn_att_embedding'))
        self.gru_layer = TimeDistributed(GRU(k2,
                                             return_state=True,
                                             return_sequences=True,
                                             # recurrent_dropout=dropout_rate,
                                             name='cnn_att_gru'))
        self.attention_feature_layer = AttentionFeatures(k1, w1, k2, w2, dropout_rate)
        self.attention_weights_layer = AttentionWeights(w3, dropout_rate)
        # dense layer: E * n_t + bias, mapped to probability of words embedding
        self.bias = self.add_weight(name='bias',
                                    shape=[vocabulary_size, ],
                                    initializer='zeros',
                                    trainable=True)
        self.softmax_layer = TimeDistributed(Softmax())
        # self.dense_layer = TimeDistributed(Dense(vocabulary_size, activation='softmax', name='cnn_att_dense'))

    def call(self, code_block: Tensor, training=False, **kwargs):
        # Note: all layers are wrapped with TimeDistributed, thus the shapes have number of
        # [batch size, timesteps (max chunk size), features (1 the subtoken value), Etc]
        # each subtoken is considered a timestep

        # create a mask of the padding sequence of the input
        mask_vector = K.cast(K.equal(code_block, 0), dtype='float32') * -1e7
        # mask_vector [batch size, max chunk length, 1]
        print("ConvAttention: mask_vector shape = {}".format(mask_vector.shape))

        # code_block = Masking(mask_value=0, )(code_block)
        tokens_embedding = self.embedding_layer(code_block)
        print("ConvAttention: Tokens shape = {}".format(tokens_embedding.shape))
        # tokens_embedding = [batch_size, max chunk length, embedding_dim]

        _, h_t = self.gru_layer(tokens_embedding, training=training)
        # h_t = [batch_size, k2)
        print("ConvAttention: h_t shape = {}".format(h_t.shape))
        l_feat = self.attention_feature_layer([tokens_embedding, h_t])
        print("ConvAttention: L_feat shape = {}".format(l_feat.shape))

        # L_feat = [batch size, max chunk size, k2]
        alpha = self.attention_weights_layer([l_feat, mask_vector])
        print("ConvAttention: alpha shape = {}".format(alpha.shape))
        # alpha = [batch size, max chunk size] weights over embeddings

        # apply the attention to the input embedding
        n_hat = K.sum((K.expand_dims(alpha, axis=-1) * tokens_embedding), axis=1)
        print("ConvAttention: n_hat shape = {}".format(n_hat.shape))
        # n_hat = [batch size, embedding dim]

        # embedding over all vocabulary
        E = self.embedding_layer.layer.embeddings
        print("ConvAttention: E shape = {}".format(E.shape))
        # E = [vocabulary size, embedding dim]

        # Apply attention to the words over all embeddings
        n_hat_E = K.nn.math_ops.tensordot(E, K.transpose(n_hat), axes=[[1], [0]])
        # n_hat_E = [vocabulary size, max chunk size, batch size]
        n_hat_E = K.permute_dimensions(n_hat_E, [2, 1, 0])
        print("ConvAttention: n_hat_E shape = {}".format(n_hat_E.shape))
        # n_hat_E = [batch size, max chunk size, vocabulary size]

        n = self.softmax_layer(K.bias_add(n_hat_E, self.bias))
        print("ConvAttention: n shape = {}".format(n.shape))
        # n = [batch size, vocabulary size] the probability of each token in the vocabulary

        return n
