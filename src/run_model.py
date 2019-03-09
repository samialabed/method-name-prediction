#!/usr/bin/env python
"""
Usage:
    train.py [options] DATA_DIR PATH_TO_CONFIG_FILE

*DATA_DIR directory filled with data with corpus extracted into .proto
*PATH_TO_CONFIG_FILE file for the model see configs/example-config.json for example

Options:
    -h --help                        Show this screen.
    --debug                          Enable debug routines. [default: False]
    --trained-model-dir=DIR          Path to a trained model weights to load and skip training.
"""
import json
import pickle
from typing import Dict

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug
from sklearn.model_selection import train_test_split

from data.preprocess import get_data_files_from_directory, PreProcessor
from models.complete_models import CnnAttentionModel

# for reproducibility
np.random.seed(1)


def run(arguments) -> None:
    config_file_path = arguments['PATH_TO_CONFIG_FILE']
    input_data_dir = arguments['DATA_DIR']

    with open(config_file_path, 'r') as fp:
        hyperparameters = json.load(fp)
    _assert_hyperparameters(hyperparameters)

    trained_model_dir = arguments.get('--trained-model-dir')

    # preprocess the data files
    datasets_preprocessors = load_train_test_validate_dataset(hyperparameters, input_data_dir)

    # TODO make this a python magic?
    if 'cnn_attention' in hyperparameters['model_type']:
        cnn_model = CnnAttentionModel(hyperparameters, datasets_preprocessors, trained_model_dir)
        print(cnn_model.evaluate_f1())


def load_train_test_validate_dataset(hyperparameters: Dict[str, any],
                                     input_data_dir: str,
                                     trained_model_path: str = None) -> Dict[str, PreProcessor]:
    preprocessor_hyperparameters = hyperparameters['preprocessor_config']
    if trained_model_path:
        with open('{}/training_data_dirs_pikls.pkl'.format(trained_model_path), 'wb') as f:
            train_data_files = pickle.load(f)

        with open('{}/testing_data_dirs_pikls.pkl'.format(trained_model_path), 'wb') as f:
            test_data_files = pickle.load(f)

        with open('{}/validating_data_dirs_pikls.pkl'.format(self.directory), 'wb') as f:
            validate_data_files = pickle.load(f)

    else:
        all_files = get_data_files_from_directory(input_data_dir,
                                                  skip_tests=preprocessor_hyperparameters['skip_tests'])
        print("Total # files: {}".format(len(all_files)))
        train_data_files, test_data_files = train_test_split(all_files, train_size=0.7)
        train_data_files, validate_data_files = train_test_split(train_data_files, train_size=0.9)
        print("Training Data: {}, Testing Data: {}, Validating data: {}".format(len(train_data_files),
                                                                                len(test_data_files),
                                                                                len(validate_data_files)))
    training_dataset_preprocessor = PreProcessor(config=preprocessor_hyperparameters,
                                                 data_files=train_data_files)
    validating_dataset_preprocessor = PreProcessor(config=preprocessor_hyperparameters,
                                                   data_files=validate_data_files,
                                                   metadata=training_dataset_preprocessor.metadata)
    testing_dataset_preprocessor = PreProcessor(config=preprocessor_hyperparameters,
                                                data_files=test_data_files,
                                                metadata=training_dataset_preprocessor.metadata)

    return {'training_dataset_preprocessor': training_dataset_preprocessor,
            'validating_dataset_preprocessor': validating_dataset_preprocessor,
            'testing_dataset_preprocessor': testing_dataset_preprocessor}


def _assert_hyperparameters(hyperparameters: Dict[str, any]):
    if 'run_name' not in hyperparameters:
        raise ValueError("No run_name given")

    if 'model_type' not in hyperparameters:
        raise ValueError("No model_type given")

    if 'model_hyperparameters' not in hyperparameters:
        raise ValueError("No model_hyperparameters given")

    # verify model hyperparameters
    model_hyperparameters = hyperparameters['model_hyperparameters']
    if 'epochs' not in model_hyperparameters:
        raise ValueError("No epochs were given in model_hyperparameters given")
    if 'batch_size' not in model_hyperparameters:
        raise ValueError("No batch_size were given in model_hyperparameters given")
    if 'max_chunk_length' not in model_hyperparameters:
        raise ValueError("No max_chunk_length were given in model_hyperparameters given")

    # verify beam search hyperparameters
    if 'beam_search_config' not in hyperparameters:
        raise ValueError("No beam_search_config were given")
    beam_search_config = hyperparameters['beam_search_config']
    if 'beam_width' not in beam_search_config:
        raise ValueError("No beam_width were given in beam_search_config given")
    if 'beam_top_paths' not in beam_search_config:
        raise ValueError("No beam_top_paths were given in beam_search_config given")

    # verify preprocessor hyperparameters
    if 'preprocessor_config' not in hyperparameters:
        raise ValueError("No preprocessor_config were given")
    preprocessor_config = hyperparameters['preprocessor_config']
    if 'vocabulary_max_size' not in preprocessor_config:
        raise ValueError("No vocabulary_max_size were given in preprocessor_config given")
    if 'max_chunk_length' not in preprocessor_config:
        raise ValueError("No max_chunk_length were given in preprocessor_config given")
    if 'vocabulary_count_threshold' not in preprocessor_config:
        raise ValueError("No vocabulary_count_threshold were given in preprocessor_config given")
    if 'min_line_of_codes' not in preprocessor_config:
        raise ValueError("No min_line_of_codes were given in preprocessor_config given")
    if 'skip_tests' not in preprocessor_config:
        raise ValueError("No skip_tests were given in preprocessor_config given")

    if model_hyperparameters['max_chunk_length'] != preprocessor_config['max_chunk_length']:
        raise ValueError("max_chunk_length differs in model_hyperparameters from preprocessor_config")


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
