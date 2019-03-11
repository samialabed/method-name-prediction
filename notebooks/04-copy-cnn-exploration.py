import logging

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import layers

from data.preprocess import PreProcessor, get_data_files_from_directory
from models.copy_cnn_attention import CopyAttention, model_objective

tf.enable_eager_execution()

hyperparameters = {
    "run_name": "copy-cnv-test",
    "model_type": "copy_attention",
    "model_hyperparameters": {
        "epochs": 10,
        "batch_size": 1,
        "k1": 32,
        "k2": 16,
        "w1": 18,
        "w2": 19,
        "w3": 2,
        "dropout_rate": 0,  # TODO make it 0.4
        "max_chunk_length": 50,
        "embedding_dim": 128,
    },
    "beam_search_config": {
        "beam_width": 5,
        "beam_top_paths": 5
    },
    "preprocessor_config": {
        "vocabulary_max_size": 5000,
        "max_chunk_length": 50,
        "vocabulary_count_threshold": 3,
        "min_line_of_codes": 3,
        "skip_tests": True
    }
}

all_files = get_data_files_from_directory(data_dir='data/raw/r252-corpus-features/org/elasticsearch/action/admin',
                                          skip_tests=hyperparameters['preprocessor_config']['skip_tests'])
print("Total # files: {}".format(len(all_files)))
train_data_files, test_data_files = train_test_split(all_files, train_size=0.7)
train_data_files, validate_data_files = train_test_split(train_data_files, train_size=0.9)
print("Training Data: {}, Testing Data: {}, Validating data: {}".format(len(train_data_files),
                                                                        len(test_data_files),
                                                                        len(validate_data_files)))
training_dataset_preprocessor = PreProcessor(config=hyperparameters['preprocessor_config'],
                                             data_files=train_data_files)
validating_dataset_preprocessor = PreProcessor(config=hyperparameters['preprocessor_config'],
                                               data_files=validate_data_files,
                                               vocabulary=training_dataset_preprocessor.vocabulary)
testing_dataset_preprocessor = PreProcessor(config=hyperparameters['preprocessor_config'],
                                            data_files=test_data_files,
                                            vocabulary=training_dataset_preprocessor.vocabulary)

# In[5]:


vocab = training_dataset_preprocessor.vocabulary
vocabulary_size = len(vocab) + 1
max_chunk_length = training_dataset_preprocessor.config['max_chunk_length']
training_data_tensors = training_dataset_preprocessor.get_tensorise_data()
testing_data_tensors = testing_dataset_preprocessor.get_tensorise_data()
validating_data_tensors = validating_dataset_preprocessor.get_tensorise_data()

# code_snippet = processed['body_tokens']
training_body_subtokens = np.expand_dims(training_data_tensors['body_tokens'], axis=-1)
training_method_name_subtokens = np.expand_dims(training_data_tensors['name_tokens'], axis=-1)

validating_dataset = (np.expand_dims(validating_data_tensors['body_tokens'], axis=-1),
                      np.expand_dims(validating_data_tensors['name_tokens'], axis=-1))

testing_dataset = (np.expand_dims(testing_data_tensors['body_tokens'], axis=-1),
                   np.expand_dims(testing_data_tensors['name_tokens'], axis=-1))

# In[20]:


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

# In[236]:

I_C = np.array([np.isin(x, y) for (x, y) in zip(training_body_subtokens, training_method_name_subtokens)])

model_hyperparameters = hyperparameters['model_hyperparameters']
model_hyperparameters["vocabulary_size"] = vocabulary_size
batch_size = model_hyperparameters['batch_size']
main_input = layers.Input(shape=(max_chunk_length, 1), batch_size=batch_size, dtype=tf.int32, name='main_input')

copy_cnn_layer = CopyAttention(model_hyperparameters)
optimizer = keras.optimizers.Nadam()  # RMSprop with Nesterov momentum

# define execution
copy_weights, n_to_map, copy_probability = copy_cnn_layer(main_input)

loss_func = model_objective(main_input, copy_probability, copy_weights)

model = keras.Model(inputs=[main_input], outputs=n_to_map)
model.compile(optimizer=optimizer,
              loss=loss_func,
              # metrics=['accuracy'],
              )

history = model.fit(training_body_subtokens,
                    training_method_name_subtokens.astype('int32'),
                    epochs=model_hyperparameters['epochs'],
                    verbose=2,
                    batch_size=batch_size,
                    # validation_data=validating_dataset,
                    )
