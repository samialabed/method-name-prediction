#!/usr/bin/env python
"""
Usage:
    run_model.py [options] DATA_DIR PATH_TO_CONFIG_FILE

*DATA_DIR directory filled with data with corpus extracted into .proto
*PATH_TO_CONFIG_FILE file for the model see configs/example-config.json for example

Options:
    -h --help                        Show this screen.
    --debug                          Enable debug routines. [default: False]
    --trained-model-dir=DIR          Path to a trained model weights to load and skip training.
    --use-same-input-dir             Use the same dataset used in the trained-model. [default: False]
"""
import json
import os
import time

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug

from models.complete_models import CnnAttentionModel
from utils.run_utils import load_train_test_validate_dataset, assert_model_hyperparameters
from utils.save_util import ReproducibilitySaver

np.random.seed(1)  # TODO remove once finished porting from old setup


def run(arguments) -> None:
    config_file_path = arguments['PATH_TO_CONFIG_FILE']
    input_data_dir = arguments['DATA_DIR']

    with open(config_file_path, 'r') as fp:
        hyperparameters = json.load(fp)
    assert_model_hyperparameters(hyperparameters)

    trained_model_dir = arguments.get('--trained-model-dir')
    restore_input_dir_used_in_training = arguments.get('--use-same-input-dir')
    if trained_model_dir:
        # TODO do I want this here?
        directory = trained_model_dir
        restore_trained_model = True
    else:
        directory = "trained_models/{}/{}/{}".format(hyperparameters['model_type'],
                                                     hyperparameters['run_name'],
                                                     time.strftime("%Y-%m-%d-%H-%M"))
        if not os.path.exists(directory):
            os.makedirs(directory)
        restore_trained_model = False  # no model to load

    reproducibility_saver = ReproducibilitySaver(directory, restore_trained_model, restore_input_dir_used_in_training)

    # preprocess the data files
    dataset_preprocessors = load_train_test_validate_dataset(hyperparameters, input_data_dir, reproducibility_saver)

    # TODO make this a python magic to automatically swap between models
    if 'cnn_attention' in hyperparameters['model_type']:
        cnn_model = CnnAttentionModel(hyperparameters, dataset_preprocessors, reproducibility_saver)
        print(cnn_model.evaluate_f1())


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['--debug']:
        import logging

        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    run_and_debug(lambda: run(args), args['--debug'])
