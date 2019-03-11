import json
import logging
import os
import pickle
from typing import Dict

import numpy as np
from dpu_utils.mlutils import Vocabulary

from data.processor import Processor


class OutputFilesNames(object):
    INPUTS_SAVE_FILE = 'inputs.txt'
    VOCABULARY_PICKLE = 'vocab.pkl'
    RANDOM_STATE_FILE = 'random.bin'
    HYPERPARAMETERS = 'hyperparameters.json'
    F1_RESULTS = 'results.txt'
    VISUALISED_INPUT_OUTPUT_FILE = 'visualised_results.txt'
    TRAINING_DATA_DIRS_PICKLE = 'training_data.pkl'
    TESTING_DATA_DIRS_PICKLE = 'testing_data.pkl'
    VALIDATING_DATA_DIRS_PICKLE = 'validating_data.pkl'
    FINAL_MODEL_WEIGHT = 'weights-final.hdf5'


class ReproducibilitySaver(object):
    # TODO is there a better pythonic way to do this?

    def __init__(self, directory: str, trained_model_dir: dir, restore_data: bool):
        self.directory = directory
        self.trained_model_dir = trained_model_dir
        self.restore_data = restore_data
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if self.trained_model_dir and self.restore_data:
            # restore saved state when restoring the model and requesting exact replica of results
            self.restore_random_state()
        elif not self.trained_model_dir:
            # new model - save the initial seed
            self.save_random_state()

    def save_random_state(self):
        self.logger.info('Saving Random State')
        with open('{}/{}'.format(self.directory, OutputFilesNames.RANDOM_STATE_FILE), 'wb') as f:
            pickle.dump(np.random.get_state(), f)

    def restore_random_state(self):
        self.logger.info('Restoring Random State')
        with open('{}/{}'.format(self.trained_model_dir, OutputFilesNames.RANDOM_STATE_FILE), 'rb') as f:
            np.random.set_state(pickle.load(f))

    def save_preprocessed_dirs(self,
                               preprocessor_object: Dict[str, Processor],
                               save_validating_file_list: bool = True,
                               save_training_file_list: bool = True,
                               save_testing_file_list: bool = True):
        # TODO make this save the tensor and not the directory

        if save_validating_file_list:
            self.logger.info('Saving Validating Data Dirs')
            with open('{}/{}'.format(self.directory, OutputFilesNames.VALIDATING_DATA_DIRS_PICKLE), 'wb') as f:
                pickle.dump(preprocessor_object['validating_dataset_preprocessor'].data_files, f)

        if save_testing_file_list:
            with open('{}/{}'.format(self.directory, OutputFilesNames.TESTING_DATA_DIRS_PICKLE), 'wb') as f:
                pickle.dump(preprocessor_object['testing_dataset_preprocessor'].data_files, f)

        if save_training_file_list:
            with open('{}/{}'.format(self.directory, OutputFilesNames.TRAINING_DATA_DIRS_PICKLE), 'wb') as f:
                pickle.dump(preprocessor_object['training_dataset_preprocessor'].data_files, f)

    def restore_preprocessed_dirs(self,
                                  restore_validating_file_list: bool = True,
                                  restore_training_file_list: bool = True,
                                  restore_testing_file_list: bool = True) -> Dict[str, np.ndarray]:
        # TODO make this restore the tensor and not the directory
        return_dir = {}
        if restore_validating_file_list:
            self.logger.info('Restoring Validating Data Dirs')
            with open('{}/{}'.format(self.trained_model_dir, OutputFilesNames.TESTING_DATA_DIRS_PICKLE), 'rb') as f:
                validating_data_files = pickle.load(f)
                return_dir['validating_data_files'] = validating_data_files
        if restore_testing_file_list:
            self.logger.info('Restoring Testing Data Dirs')
            with open('{}/{}'.format(self.trained_model_dir, OutputFilesNames.TESTING_DATA_DIRS_PICKLE), 'rb') as f:
                testing_data_files = pickle.load(f)
                return_dir['testing_data_files'] = testing_data_files

        if restore_training_file_list:
            self.logger.info('Restoring Training Data Dirs')
            with open('{}/{}'.format(self.trained_model_dir, OutputFilesNames.TRAINING_DATA_DIRS_PICKLE), 'rb') as f:
                training_data_files = pickle.load(f)
                return_dir['training_data_files'] = training_data_files

        return return_dir

    def save_vocabulary(self, vocabulary):
        self.logger.info("Saving trained model vocabulary")
        with open('{}/{}'.format(self.directory, OutputFilesNames.VOCABULARY_PICKLE), 'wb') as f:
            pickle.dump(vocabulary, f)

    def restore_vocabulary(self) -> Vocabulary:
        self.logger.info("Restoring trained model vocabulary")
        with open('{}/{}'.format(self.trained_model_dir, OutputFilesNames.VOCABULARY_PICKLE), 'rb') as f:
            vocabulary = pickle.load(f)
        return vocabulary

    def save_into_input_info_file(self, message):
        with open('{}/{}'.format(self.directory, OutputFilesNames.INPUTS_SAVE_FILE), 'a') as fp:
            inputs_str = "{}{}".format(message, os.linesep)
            fp.write(inputs_str)

    def save_visualised_results(self, visualised_input):
        with open('{}/{}'.format(self.directory, OutputFilesNames.VISUALISED_INPUT_OUTPUT_FILE), 'w') as fp:
            fp.write(visualised_input)

    def save_f1_results(self, f1_evaluation):
        with open('{}/{}'.format(self.directory, OutputFilesNames.F1_RESULTS), 'w') as fp:
            fp.write(str(f1_evaluation))

    def save_hyperparameters(self, hyperparameters):
        with open('{}/{}'.format(self.directory, OutputFilesNames.HYPERPARAMETERS), 'w') as fp:
            json.dump(hyperparameters, fp)

    def restore_hyperparameters(self) -> Dict[str, any]:
        with open('{}/{}'.format(self.trained_model_dir, OutputFilesNames.HYPERPARAMETERS), 'r') as fp:
            hyperparameters = json.load(fp)
        return hyperparameters
