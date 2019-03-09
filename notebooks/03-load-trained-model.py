import json

import numpy as np

from models.complete_models import CnnAttentionModel
from run_model import load_train_test_validate_dataset

np.random.seed(1)
config_file_path = 'configs/example-config.json'
input_data_dir = 'data/raw/r252-corpus-features/org/elasticsearch/action/admin'
trained_model_path = 'trained_models/cnn_attention/elasticsearch_small_overfit_tests/2019-03-09-14-54'
with open(config_file_path, 'r') as fp:
    hyperparameters = json.load(fp)

datasets_preprocessors = load_train_test_validate_dataset(hyperparameters, input_data_dir)

cnn_model = CnnAttentionModel(hyperparameters, datasets_preprocessors, trained_model_path)

cnn_model.evaluate_f1()
