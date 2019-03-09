import logging
from typing import Dict

import numpy as np
from tensorflow.python import keras, Tensor
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, GRU, TimeDistributed, Softmax, Conv1D, MaxPooling1D

from models.attention import AttentionFeatures, AttentionWeights


class CopyAttention(keras.Model):
    """
    <From the paper>
    extends the CNN-attention with a copy mmechanismthat allows it to suggest out of vocabulary subtokens.
    """

    def __init__(self, hyperparameters: Dict[str, any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
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
        self.attention_weights_alpha_layer = AttentionWeights(w3, dropout_rate)
        self.attention_weights_k_layer = AttentionWeights(w3, dropout_rate)
        self.lambda_conv_layer = TimeDistributed(Conv1D(1, w3, activation='sigmoid'))
        self.max_layer = TimeDistributed(MaxPooling1D(1))
        # dense layer: E * n_t + bias, mapped to probability of words embedding
        self.bias = self.add_weight(name='bias',
                                    shape=[vocabulary_size, ],
                                    initializer='zeros',
                                    trainable=True)
        self.softmax_layer = TimeDistributed(Softmax())

    def call(self, code_block: Tensor, training=False, **kwargs):
        # Note: all layers are wrapped with TimeDistributed, thus the shapes have number of
        # [batch size, timesteps (token length), features (1 the subtoken value), Etc]
        # each subtoken is considered a timestep

        # create a mask of the padding sequence of the input
        mask_vector = K.cast(K.equal(code_block, 0), dtype='float32') * -1e7
        # mask_vector [batch size, max chunk length, 1]
        self.logger.info("mask_vector shape = {}".format(mask_vector.shape))

        # code_block = Masking(mask_value=0, )(code_block)
        tokens_embedding = self.embedding_layer(code_block)
        self.logger.info("Tokens shape = {}".format(tokens_embedding.shape))
        # tokens_embedding = [batch_size, max chunk length, embedding_dim]

        _, h_t = self.gru_layer(tokens_embedding, training=training)
        # h_t = [batch_size, k2)
        self.logger.info("h_t shape = {}".format(h_t.shape))
        l_feat = self.attention_feature_layer([tokens_embedding, h_t])
        self.logger.info("L_feat shape = {}".format(l_feat.shape))

        # L_feat = [batch size, token length, k2]
        alpha = self.attention_weights_alpha_layer([l_feat, mask_vector])
        self.logger.info("alpha shape = {}".format(alpha.shape))
        # alpha = [batch size, token length] weights over embeddings

        # apply the attention to the input embedding
        n_hat = K.sum((K.expand_dims(alpha, axis=-1) * tokens_embedding), axis=1)
        self.logger.info("n_hat shape = {}".format(n_hat.shape))
        # n_hat = [batch size, embedding dim]

        # embedding over all vocabulary
        E = self.embedding_layer.layer.embeddings
        self.logger.info("E shape = {}".format(E.shape))
        # E = [vocabulary size, embedding dim]

        # Apply attention to the words over all embeddings
        n_hat_E = K.nn.math_ops.tensordot(E, K.transpose(n_hat), axes=[[1], [0]])
        # n_hat_E = [vocabulary size, token length, batch size]
        n_hat_E = K.permute_dimensions(n_hat_E, [2, 1, 0])
        self.logger.info("n_hat_E shape = {}".format(n_hat_E.shape))
        # n_hat_E = [batch size, token length, vocabulary size]

        n = self.softmax_layer(K.bias_add(n_hat_E, self.bias))
        self.logger.info("n shape = {}".format(n.shape))
        # n = [batch size, vocabulary size] the probability of each token in the vocabulary

        # copy_attention extension
        k_layer = self.attention_weights_alpha_layer([l_feat, mask_vector])
        print("Copy_CNN_attention: k_layer shape: {}".format(k_layer.shape))
        lmda = self.max_layer(self.lambda_conv_layer(l_feat))
        print("Copy_CNN_attention: lmda shape: {}".format(lmda.shape))

        return (lmda * k_layer), ((1 - lmda) * n), lmda


# TODO move tihs somewhere else
def model_objective(code_subtoken, target_subtoken, lmda, kappa):
    # vector of a target subtoken exist in the input token
    I_C = np.isin(code_subtoken, target_subtoken).astype(int)
    probability_correct_copy = lmda * np.sum(I_C * kappa)

    meu = -1e10

    #
    # def lossFunction(y_true, y_pred):
    #     loss = mse(y_true, y_pred)
    #     loss += K.sum(val, K.abs(K.sum(K.square(layer_weights), axis=1)))
    #     return loss

    # return lossFunction
