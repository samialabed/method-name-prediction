from typing import Dict

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data.processor import Processor, get_data_files_from_directory
from utils.save_util import ReproducibilitySaver


def save_train_validate_history(directory: str, history):
    # TODO move it to ReproducibilitySaver
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}/model_accuracy.png'.format(directory))
    plt.clf()  # Clear the figure for the next loop

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}/model_loss.png'.format(directory))


def load_train_test_validate_dataset(hyperparameters: Dict[str, any],
                                     input_data_dir: str,
                                     reproducibility_saver: ReproducibilitySaver) -> Dict[str, Processor]:
    preprocessor_hyperparameters = hyperparameters['preprocessor_config']

    vocabulary = None
    # TODO make it save the tensorised value
    if reproducibility_saver.trained_model_dir:
        vocabulary = reproducibility_saver.restore_vocabulary()

    if reproducibility_saver.restore_data:
        restored_dirs = reproducibility_saver.restore_preprocessed_dirs()
        train_data_files = restored_dirs['training_data_files']
        test_data_files = restored_dirs['testing_data_files']
        validate_data_files = restored_dirs['validating_data_files']
    else:
        print("Manually loading files from input_data_dir")
        all_files = get_data_files_from_directory(input_data_dir,
                                                  skip_tests=preprocessor_hyperparameters['skip_tests'])
        print("Total # files: {}".format(len(all_files)))
        train_data_files, test_data_files = train_test_split(all_files, train_size=0.7)
        train_data_files, validate_data_files = train_test_split(train_data_files, train_size=0.9)
        print("Training Data: {}, Testing Data: {}, Validating data: {}".format(len(train_data_files),
                                                                                len(test_data_files),
                                                                                len(validate_data_files)))
    training_dataset_preprocessor = Processor(config=preprocessor_hyperparameters,
                                              data_files=train_data_files,
                                              vocabulary=vocabulary)
    validating_dataset_preprocessor = Processor(config=preprocessor_hyperparameters,
                                                data_files=validate_data_files,
                                                vocabulary=training_dataset_preprocessor.vocabulary)
    testing_dataset_preprocessor = Processor(config=preprocessor_hyperparameters,
                                             data_files=test_data_files,
                                             vocabulary=training_dataset_preprocessor.vocabulary)

    return {'training_dataset_preprocessor': training_dataset_preprocessor,
            'validating_dataset_preprocessor': validating_dataset_preprocessor,
            'testing_dataset_preprocessor': testing_dataset_preprocessor}


def assert_model_hyperparameters(hyperparameters: Dict[str, any]):
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
