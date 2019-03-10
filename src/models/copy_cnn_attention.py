import logging
from typing import Dict

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
        self.attention_weights_kappa_layer = AttentionWeights(w3, dropout_rate)
        self.lambda_conv_layer = TimeDistributed(Conv1D(1, w3, activation='sigmoid'))
        self.max_layer = TimeDistributed(MaxPooling1D(pool_size=1, strides=50))
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
        self.logger.info("Copy_CNN_attention: n shape: {}".format(n.shape))

        # copy_attention extension
        kappa = self.attention_weights_kappa_layer([l_feat, mask_vector])
        self.logger.info("kappa shape: {}".format(kappa.shape))
        # kappa = [batch size, token length] weights over embeddings

        # lmda = probability to copy from the copy conv
        lmda = K.squeeze(self.max_layer(self.lambda_conv_layer(l_feat)), axis=-1)
        self.logger.info("lmda shape: {}".format(lmda.shape))

        # pos2voc = probability of subtokens assigned to the copy mechanism kappa, effectively acting as copy weight
        pos2voc = K.sum((K.expand_dims(kappa, axis=-1) * tokens_embedding), axis=1)
        self.logger.info("pos2voc shape: {}".format(pos2voc.shape))
        # pos2voc = [batch size, body length, embed dim]

        # Make sure the shape doesn't change
        weighted_n = (1 - lmda) * n
        self.logger.info("weighted_n shape:{}".format(weighted_n.shape))
        weighted_pos2voc = lmda * pos2voc
        self.logger.info("weighted_pos2voc shape:{}".format(weighted_pos2voc.shape))

        return weighted_pos2voc, weighted_n, lmda


def model_objective(input_code_subtoken, copy_probability, copy_weights):
    # copy_weights = lambda in the paper
    # copy_probability = kappa
    # input_code_subtoken = c
    print("Model objective: input_code_subtoken.shape: {}".format(input_code_subtoken.shape))
    print("Model objective: copy_probability.shape: {}".format(copy_probability.shape))
    print("Model objective: copy_weights.shape: {}".format(copy_weights.shape))

    unknown_id = 1  # TODO move this to be fed at input time Vocab.get_id_or_ukno()
    mu = -10e-8  # TODO take it as hyperparameter

    # TODO consider using log on your values
    def loss_function(target_subtoken, y_pred):
        # prediction is a probability, log probability for speed and smoothness

        print("Model objective: y_pred.shape: {}".format(y_pred.shape))
        # I_C = vector of a target subtoken exist in the input token - TODO probably not ok, debug using TF eager
        I_C = K.expand_dims(K.cast(K.any(K.equal(input_code_subtoken,
                                                 K.cast(target_subtoken, 'int32')),
                                         axis=-1), dtype='float32'), -1)
        print("Model objective: I_C.shape: {}".format(I_C.shape))
        # I_C shape = [batch_size, token, max_char_len, 1]
        # TODO should I add a penality if there is no subtokens appearing in the model ? Yes
        probability_correct_copy = K.log(copy_probability) + K.log(K.sum(I_C * copy_weights) + mu)
        print("Model objective: probability_correct_copy.shape: {}".format(probability_correct_copy.shape))

        # penalise the model when cnn-attention predicts unknown
        # but the value can be predicted from the copy mechanism.
        mask_unknown = K.cast(K.equal(target_subtoken, unknown_id), dtype='float32') * mu

        probability_target_token = K.sum(K.log(1 - copy_probability) + K.log(y_pred) + mask_unknown, -1, True)
        print("Model objective: probability_target_token.shape: {}".format(probability_target_token.shape))

        loss = K.logsumexp([probability_correct_copy, probability_target_token])
        return K.mean(loss)

    return loss_function
