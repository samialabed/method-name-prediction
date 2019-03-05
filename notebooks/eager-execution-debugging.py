import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from data.preprocess import PreProcessor
from data.utils import predict_name
from models.cnn_attention import ConvAttention

tf.enable_eager_execution()

data = PreProcessor(config=PreProcessor.DEFAULT_CONFIG,
                    data_dir='data/raw/r252-corpus-features/org/elasticsearch/action/admin/cluster/allocation/')

vocab = data.metadata['token_vocab']
processed = data.get_tensorise_data()

vocabulary_size = len(vocab) + 1
max_chunk_length = data.config['max_chunk_length']
code_snippet = np.expand_dims(processed['body_tokens'], -1)
label_name = np.expand_dims(processed['name_tokens'], axis=-1)

print("Vocab Size: {} number of Code snippet: {} number of labels: {}".format(vocabulary_size, len(code_snippet),
                                                                              len(label_name)))
print("Label_name shape: {}\nCode_snippet shape: {}".format(label_name.shape, code_snippet.shape))

# TODO make the input a json file and parse it
hyperparameter = {'batch_size': 1, 'k1': 8, 'k2': 8, 'w1': 24, 'w2': 29, 'w3': 10, 'dropout_rate': 0.5,
                  'max_chunk_length': max_chunk_length, 'vocabulary_size': vocabulary_size, 'embedding_dim': 128}
# Optimised hyperparameter are reported in page 5 of the paper

batch_size = hyperparameter['batch_size']
# define layers
main_input = layers.Input(shape=(max_chunk_length, 1),
                          batch_size=batch_size,
                          dtype=tf.int32, name='main_input',
                          )

cnn_layer = ConvAttention(hyperparameter)

optimizer = keras.optimizers.Nadam()  # RMSprop with Nesterov momentum
# loss_func = masked_sparse_cross_entropy_loss
loss_func = keras.losses.sparse_categorical_crossentropy

# define execution
cnn_output = cnn_layer(main_input)
model = keras.Model(inputs=[main_input], outputs=cnn_output)
model.compile(optimizer=optimizer,
              loss=loss_func,
              metrics=['accuracy'],
              )
# fit the model

dataset = tf.data.Dataset.from_tensor_slices((code_snippet, label_name))
dataset = dataset.shuffle(1000).batch(1)

history = model.fit(dataset,
                    # label_name,
                    epochs=27,
                    verbose=2,
                    batch_size=batch_size,
                    steps_per_epoch=1
                    )

for images, labels in dataset.take(1):
    print("Logits: ", model(images[0:1]).numpy())

#
#
# model.load_weights('model.h5')
#
#
# for (batch, (cd_block, meth_name)) in enumerate(dataset.take(1)):
#     test = model.predict(cd_block.numpy()).argmax(-1)
# test = predict_name(vocab, model, cd_block.numpy())
# print(model.predict(cd_block.numpy()))
# print(test)

# translate prediction
