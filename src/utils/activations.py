from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import nn


class LogSoftmax(Layer):
    """LogSoftmax activation function.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Arguments:
        axis: Integer, axis along which the LogSoftmax normalization is applied.
    """

    def __init__(self, axis=-1, **kwargs):
        super(LogSoftmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return nn.log_softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(LogSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


def masked_sparse_cross_entropy_loss(yTrue, yPred):
    # find which values in yTrue (target) are the mask value
    isMask = K.equal(yTrue, 0)  # true for all mask values

    # since y is shaped as (batch, length, features), we need all features to be mask values
    isMask = K.all(isMask, axis=-1)  # the entire output vector must be true
    # this second line is only necessary if the output features are more than 1

    # transform to float (0 or 1) and invert
    isMask = K.cast(isMask, dtype=K.floatx())
    isMask = 1 - isMask  # now mask values are zero, and others are 1

    # multiply this by the inputs:
    # maybe you might need K.expand_dims(isMask) to add the extra dimension removed by K.all
    yTrue = yTrue * isMask
    yPred = yPred * isMask
    print("yPred: {} \nyTrue: {}".format(yPred, yTrue))
    return K.sparse_categorical_crossentropy(yTrue, yPred)
