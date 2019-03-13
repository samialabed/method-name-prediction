import logging
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from dpu_utils.mlutils import Vocabulary
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data.processor import Processor
from models.cnn_attention import ConvAttention
from utils.f1_evaluator import evaluate_f1
from utils.run_utils import save_train_validate_history
from utils.save_util import ReproducibilitySaver, OutputFilesNames


class CnnAttentionModel(object):
    def __init__(self,
                 hyperparameters: Dict[str, any],
                 preprocessors: Dict[str, Union[Processor, Vocabulary]],
                 reproducibility_saver: ReproducibilitySaver):
        self.reproducibility_saver = reproducibility_saver
        self.hyperparameters = hyperparameters
        self.preprocessors = preprocessors
        self.vocab = preprocessors['vocabulary']
        self.logger = logging.getLogger(__name__)
        self.directory = self.reproducibility_saver.directory

        # create model
        self.model = self._compile_cnn_attention_model()

        if self.reproducibility_saver.trained_model_dir:
            self.logger.info('Loading saved weights')
            self.model.load_weights("{}/{}".format(self.reproducibility_saver.trained_model_dir,
                                                   OutputFilesNames.FINAL_MODEL_WEIGHT))
        else:
            # Save name of files to allow reproducibility
            self.logger.info('Saving hyperparameters, training, testing, validating, and vocabs')
            self.reproducibility_saver.save_hyperparameters(hyperparameters)
            self.reproducibility_saver.save_preprocessed_dirs(preprocessors)
            self.reproducibility_saver.save_vocabulary(self.vocab)
            self._train_cnn_attention_model()

    def evaluate_f1(self):
        # testing loop
        testing_data_tensors = self.preprocessors['testing_dataset_preprocessor'].get_tensorise_data()
        testing_body_subtokens = np.expand_dims(testing_data_tensors['body_tokens'], axis=-1)
        testing_method_name_subtokens = np.expand_dims(testing_data_tensors['name_tokens'], axis=-1)
        self.logger.info('Evaluate F1 score on corpus {}'.format(testing_body_subtokens.shape[0]))
        f1_evaluation, visualised_input = evaluate_f1(self.model,
                                                      self.vocab,
                                                      testing_body_subtokens,
                                                      testing_method_name_subtokens,
                                                      self.hyperparameters['beam_search_config'],
                                                      visualise_prediction=True)
        self.reproducibility_saver.save_f1_results(f1_evaluation)
        self.reproducibility_saver.save_visualised_results(visualised_input)
        self.reproducibility_saver.save_into_input_info_file(testing_body_subtokens.shape[0])

        return f1_evaluation

    def _compile_cnn_attention_model(self):
        model_hyperparameters = self.hyperparameters['model_hyperparameters']
        model_hyperparameters["vocabulary_size"] = len(self.vocab) + 1
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

    def _train_cnn_attention_model(self):
        # get the data and curate it for the model
        training_data_tensors = self.preprocessors['training_dataset_preprocessor'].get_tensorise_data()
        validating_data_tensors = self.preprocessors['validating_dataset_preprocessor'].get_tensorise_data()

        # get tensorised training/validating dataset
        training_body_subtokens = np.expand_dims(training_data_tensors['body_tokens'], axis=-1)
        training_method_name_subtokens = np.expand_dims(training_data_tensors['name_tokens'], axis=-1)

        validating_dataset = (np.expand_dims(validating_data_tensors['body_tokens'], axis=-1),
                              np.expand_dims(validating_data_tensors['name_tokens'], axis=-1))

        input_information = "Training samples: {}, validating samples: {}".format(training_body_subtokens.shape[0],
                                                                                  validating_dataset[0].shape[0])
        self.reproducibility_saver.save_into_input_info_file(input_information)

        # training loop
        model_hyperparameters = self.hyperparameters['model_hyperparameters']
        checkpoint_fp = "{}/weights-{{epoch:02d}}-{{val_acc:.2f}}.hdf5".format(self.directory)
        checkpoint = ModelCheckpoint(checkpoint_fp, monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        history = self.model.fit(training_body_subtokens,
                                 training_method_name_subtokens,
                                 epochs=model_hyperparameters['epochs'],
                                 verbose=2,
                                 batch_size=model_hyperparameters['batch_size'],
                                 callbacks=callbacks_list,
                                 validation_data=validating_dataset,
                                 )
        self.model.save_weights("{}/weights-final.hdf5".format(self.directory))
        save_train_validate_history(self.directory, history)
