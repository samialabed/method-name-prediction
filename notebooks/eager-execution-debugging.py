import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from data.preprocess import PreProcessor
from data.utils import translate_tokenized_array_to_list_words, predict_name
from models.cnn_attention import ConvAttention
from utils.activations import masked_sparse_cross_entropy_loss

tf.enable_eager_execution()

# data = PreProcessor(config=PreProcessor.DEFAULT_CONFIG,
#                     data_dir='data/raw/r252-corpus-features/org/elasticsearch/action/admin/')
data = PreProcessor(config=PreProcessor.DEFAULT_CONFIG,
                    data_dir='data/raw/r252-corpus-features/org/elasticsearch/action/admin/')

vocab = data.metadata['token_vocab']
processed = data.get_tensorise_data()

embedding_dim = 128
vocabulary_size = len(vocab) + 1
max_chunk_length = data.config['max_chunk_length']
code_snippet = processed['body_tokens']
# code_snippet = np.expand_dims(processed['body_tokens'], 0)
# label_name = np.expand_dims(processed['name_tokens'], -1)
label_name = processed['name_tokens']
# label_name = keras.utils.to_categorical(processed['name_tokens'], num_classes=vocabulary_size)
print("Vocab Size: {} Code snippet len: {} label_name len: {}".format(vocabulary_size, len(code_snippet),
                                                                      len(label_name)))

# TODO make the input a json file and parse it
batch_size = 1
k1 = 8
k2 = 8
w1 = 24
w2 = 29
w3 = 10
dropout_rate = 0.5

# Optimised hyperparameter are reported in page 5 of the paper

# define layers
main_input = layers.Input(shape=(max_chunk_length,),
                          batch_size=batch_size,
                          dtype=tf.int32, name='main_input')
hyperparameter = {'batch_size': 1, 'k1': 8, 'k2': 8, 'w1': 24, 'w2': 29, 'w3': 10, 'dropout_rate': 0.5,
                  'max_chunk_length': max_chunk_length, 'vocabulary_size': vocabulary_size, 'embedding_dim': 128}
cnn_layer = ConvAttention(hyperparameter)

optimizer = keras.optimizers.Nadam()  # RMSprop with Nesterov momentum
loss_func = masked_sparse_cross_entropy_loss
# loss_func =  keras.losses.sparse_categorical_crossentropy

# define execution
cnn_output = cnn_layer(main_input)
model = keras.Model(inputs=[main_input], outputs=cnn_output)
model.compile(optimizer=optimizer,
              loss=loss_func,
              # metrics=['sparse_categorical_accuracy'],
              )
# fit the model
# tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
# history = model.fit(code_snippet,
#                     label_name,
#                     epochs=10,
#                     verbose=2,
#                     batch_size=batch_size,
#                     callbacks=[tbCallBack],
#                     validation_split=0.2)

model.load_weights('model.h5')

code_snippet = processed['body_tokens']
label_name = processed['name_tokens']
dataset = tf.data.Dataset.from_tensor_slices((code_snippet, label_name))
dataset = dataset.shuffle(1000).batch(1)

for (batch, (cd_block, meth_name)) in enumerate(dataset.take(1)):
    test = predict_name(vocab, model, cd_block.numpy())
    # print(model.predict(cd_block.numpy()))
    print(test)

# translate prediction