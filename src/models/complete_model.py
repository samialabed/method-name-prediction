import json
import os
import time
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data.preprocess import PreProcessor
from models.cnn_attention import ConvAttention
from utils.f1_evaluator import evaluate_f1
from utils.run_utils import save_train_validate_history


def run_cnn_attention_model(hyperparameters: Dict[str, any], preprocessors: Dict[str, PreProcessor]):
    # Optimised hyperparameter are reported in page 5 of the paper
    # checkpointing - TODO move it to util
    directory = "trained_models/{}/{}/{}".format(hyperparameters['model_type'],
                                                 hyperparameters['run_name'],
                                                 time.strftime("%Y-%m-%d-%H-%M"))
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = "{}/weights-{{epoch:02d}}-{{val_acc:.2f}}.hdf5".format(directory)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    with open('{}/config.json'.format(directory), 'w') as fp:
        json.dump(hyperparameters, fp)

    training_data_tensors = preprocessors['training_dataset_preprocessor'].get_tensorise_data()
    validating_data_tensors = preprocessors['validating_dataset_preprocessor'].get_tensorise_data()

    # get tensorised training/validating dataset
    training_body_subtokens = np.expand_dims(training_data_tensors['body_tokens'], axis=-1)
    training_method_name_subtokens = np.expand_dims(training_data_tensors['name_tokens'], axis=-1)

    validating_dataset = (np.expand_dims(validating_data_tensors['body_tokens'], axis=-1),
                          np.expand_dims(validating_data_tensors['name_tokens'], axis=-1))

    vocab = preprocessors['training_dataset_preprocessor'].metadata['token_vocab']

    # create model
    model_hyperparameters: Dict[str, any] = hyperparameters['model_hyperparameters']
    model_hyperparameters["vocabulary_size"] = len(vocab) + 1
    model = compile_cnn_attention_model(model_hyperparameters)
    # train loop
    history = model.fit(training_body_subtokens,
                        training_method_name_subtokens,
                        epochs=model_hyperparameters['epochs'],
                        verbose=2,
                        batch_size=model_hyperparameters['batch_size'],
                        callbacks=callbacks_list,
                        validation_data=validating_dataset,
                        )

    save_train_validate_history(directory, history)

    # testing loop
    testing_data_tensors = preprocessors['testing_dataset_preprocessor'].get_tensorise_data()
    testing_body_subtokens = np.expand_dims(testing_data_tensors['body_tokens'], axis=-1)
    testing_method_name_subtokens = np.expand_dims(testing_data_tensors['name_tokens'], axis=-1)

    f1_evaluation = evaluate_f1(model, vocab, testing_body_subtokens, testing_method_name_subtokens,
                                hyperparameters['beam_search_config'])
    with open('{}/results.txt'.format(directory), 'w') as fp:
        fp.write(str(f1_evaluation))


def compile_cnn_attention_model(model_hyperparameters: Dict[str, any]):
    batch_size = model_hyperparameters['batch_size']
    main_input = layers.Input(shape=(None, 1), batch_size=batch_size, dtype=tf.int32, name='main_input')
    cnn_layer = ConvAttention(model_hyperparameters)
    optimizer = keras.optimizers.Nadam()  # RMSprop with Nesterov momentum
    loss_func = keras.losses.sparse_categorical_crossentropy
    # define execution
    cnn_output = cnn_layer(main_input)
    model = keras.Model(inputs=[main_input], outputs=cnn_output)
    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['accuracy'])
    return model
