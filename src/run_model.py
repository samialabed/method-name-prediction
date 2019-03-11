#!/usr/bin/env python
"""
Usage:
    run_model.py DATA_DIR (--hyperparameters-config=FILE | --trained-model-dir=DIR [--use-same-input-dir]) [options]

Executes the model defined in the config file or the trained model hyperparameters.

* DATA_DIR directory filled with data with corpus extracted into .proto
* --hyperparameters-config=FILE    PATH file for the model hyperparameters. see configs/example-config.json.
* --trained-model-dir=DIR          Path to a trained model directory to skip training and restore vocabulary.
* --use-same-input-dir             Use the same dataset used in the trained-model. [default: False]

Must choose between either passing a hyperparameters thus training a new model or passing a previously trained model
and retrieving its hyperparameters.

Options:
    -h --help                        Show this screen.
    --debug                          Enable debug routines. [default: False]
"""
import json
import time

from docopt import docopt
from dpu_utils.utils import run_and_debug

from models.complete_models import CnnAttentionModel
from utils.run_utils import load_train_test_validate_dataset, assert_model_hyperparameters
from utils.save_util import ReproducibilitySaver


def run(arguments) -> None:
    input_data_dir = arguments['DATA_DIR']

    config_file_path = arguments.get('--hyperparameters-config')
    trained_model_dir = arguments.get('--trained-model-dir')
    restore_inputs_used_in_training = arguments.get('--use-same-input-dir')

    if config_file_path:
        with open(config_file_path, 'r') as fp:
            hyperparameters = json.load(fp)
        assert_model_hyperparameters(hyperparameters)
        directory = "trained_models/{}/{}/{}".format(hyperparameters['model_type'],
                                                     hyperparameters['run_name'],
                                                     time.strftime("%Y-%m-%d-%H-%M"))
        reproducibility_saver = ReproducibilitySaver(directory, None, False)

    else:
        # Start a sub directory to put all new experiments that are made on top of the pre-existence model.
        directory = "{}/experiments/{}/".format(trained_model_dir,
                                                time.strftime("%Y-%m-%d-%H-%M"))

        # pass the trained model to restore states from it
        reproducibility_saver = ReproducibilitySaver(directory, trained_model_dir, restore_inputs_used_in_training)
        hyperparameters = reproducibility_saver.restore_hyperparameters()

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
