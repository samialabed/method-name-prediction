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
    --use-same-input-dir             Use the same datasets for trianing/validating/testing as the trained-model
"""
import json

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug

from models.complete_models import CnnAttentionModel
# for reproducibility
from utils.run_utils import load_train_test_validate_dataset, _assert_hyperparameters

np.random.seed(1)


def run(arguments) -> None:
    config_file_path = arguments['PATH_TO_CONFIG_FILE']
    input_data_dir = arguments['DATA_DIR']

    with open(config_file_path, 'r') as fp:
        hyperparameters = json.load(fp)
    _assert_hyperparameters(hyperparameters)

    # TODO give the choice to skip using previously defined training data
    trained_model_dir = arguments.get('--trained-model-dir')
    use_same_input_as_trained_model = arguments.get('--use-same-input-dir')

    # preprocess the data files
    datasets_preprocessors = load_train_test_validate_dataset(hyperparameters, input_data_dir, trained_model_dir,
                                                              use_same_input_as_trained_model)

    # TODO make this a python magic?
    if 'cnn_attention' in hyperparameters['model_type']:
        cnn_model = CnnAttentionModel(hyperparameters, datasets_preprocessors, trained_model_dir)
        print(cnn_model.evaluate_f1())


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
