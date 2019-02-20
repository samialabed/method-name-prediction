import logging
import os
from collections import Counter
from glob import iglob
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np
from dpu_utils.mlutils import Vocabulary

from data.graph_feature_extractor import GraphFeatureExtractor
from data.graph_pb2 import Graph

NameBodyTokens = Tuple[List[str], List[str]]
LoadedSamples = Dict[str, np.ndarray]
DATA_FILE_EXTENSION = 'proto'


class PreProcessor(object):
    DEFAULT_CONFIG = {
        'vocabulary_max_size': 5000,  # the vocabulary embedding maximum size.
        'max_chunk_length': 50,  # the maximum size of a token, smaller tokens will be padded to size.
        'vocabulary_count_threshold': 2,  # the minimum occurrences of a token to not be considered a rare token.
        'run_name': 'default_parser',  # meaningful name of the experiment configuration.
        'min_line_of_codes': 3,  # minimum line of codes the method should contain to be considered in the corpus.
        'skip_tests': True  # skip files that contain test
    }

    def __init__(self, config: Dict[str, Any], data_dir: str = 'data/raw/r252-corpus-features/',
                 max_num_files: int = None, metadata: Dict[str, Any] = None):
        """
        :param config: dictionary containing parsers configs and vocabulary size.
        :param data_dir: path to data input directory
        :param max_num_files: Maximal number of files to load.
        :param metadata: (Optional) metadata about the corpus, holds vocabulary. This is useful for test dataset.
        """
        if config is None:
            config = self.DEFAULT_CONFIG
        self.config = config
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.max_num_files = max_num_files
        self.data_files = self.load_data_files_from_directory()
        self.corpus_methods_token = self.get_tokens_from_dir()
        if metadata is None:
            metadata = self.load_metadata()
        self.metadata = metadata

    def load_metadata(self) -> Dict[str, Vocabulary]:
        """ Return model metadata such as a vocabulary. """
        max_size = self.config['vocabulary_max_size']
        count_threshold = self.config['vocabulary_count_threshold']
        # Count occurrences of the body vocabulary
        tokens_counter = Counter()

        for method_token in self.corpus_methods_token:
            for (name, body) in method_token:
                tokens_counter.update(body)
                tokens_counter.update(name)

        token_vocab = Vocabulary.create_vocabulary(tokens_counter,
                                                   count_threshold=count_threshold,
                                                   max_size=max_size,
                                                   add_unk=True,
                                                   add_pad=True)

        self.logger.info('{} Vocabulary created'.format(len(token_vocab)))
        # TODO - add more stats about the directory, such as number of methods, longest method, etc.
        return {'token_vocab': token_vocab}

    def get_tensorise_data(self) -> LoadedSamples:
        """ Returns a tensoirsed data representation from directory path"""
        return self.load_data_from_raw_sample_sequences(token_seq for token_seq in self.corpus_methods_token)

    def load_data_from_raw_sample_sequences(self, files_token_seqs: Iterable[List[NameBodyTokens]]) -> LoadedSamples:
        """
        Load and tensorise data from a file.
        :param files_token_seqs: Sequences of tokens per file to load samples from.
        :return The loaded data, as a dictionary mapping names to numpy arrays.
        """
        loaded_data = {'name_tokens': [], 'name_tokens_length': [], 'body_tokens': [], 'body_tokens_lengths': []}

        max_chunk_length = self.config['max_chunk_length']
        vocab = self.metadata['token_vocab']

        for file_token_seqs in files_token_seqs:
            for (method_name, method_body) in file_token_seqs:
                loaded_data['name_tokens'].append(vocab.get_id_or_unk_multiple(method_name,
                                                                               pad_to_size=max_chunk_length))
                loaded_data['name_tokens_length'].append(len(method_name))
                loaded_data['body_tokens_lengths'].append(len(method_body))
                loaded_data['body_tokens'].append(vocab.get_id_or_unk_multiple(method_body,
                                                                               pad_to_size=max_chunk_length))

        # Turn into numpy arrays for easier slicing later:
        assert len(loaded_data['body_tokens']) == len(loaded_data['body_tokens_lengths']), \
            "Loaded 'body_tokens' and 'body_tokens_lengths' lists need to be aligned and of" \
            + "the same length!"

        assert len(loaded_data['name_tokens']) == len(loaded_data['name_tokens_length']), \
            "Loaded 'name_tokens' and 'name_tokens_length' lists need to be aligned and of" \
            + "the same length!"

        loaded_data['name_tokens'] = np.array(loaded_data['name_tokens'])
        loaded_data['name_tokens_length'] = np.array(loaded_data['name_tokens_length'])

        loaded_data['body_tokens'] = np.array(loaded_data['body_tokens'])
        loaded_data['body_tokens_lengths'] = np.array(loaded_data['body_tokens_lengths'])

        return loaded_data

    def get_tokens_from_dir(self) -> List[List[NameBodyTokens]]:
        """ Returns a list of all tokens in the data files. """
        return [methods_token for file in self.data_files for methods_token in self.load_data_file(file)]

    def load_data_files_from_directory(self) -> List[str]:
        files = iglob(os.path.join(self.data_dir, '**/*.{}'.format(DATA_FILE_EXTENSION)), recursive=True)

        # Skip tests and exception classes
        if self.config['skip_tests']:
            files = filter(lambda file: not file.endswith(("Test.java.proto", "Exception*")), files)
        if self.max_num_files:
            files = sorted(files)[:int(self.max_num_files)]
        else:
            files = list(files)
        np.random.shuffle(files)
        return files

    def load_data_file(self, path: str) -> Iterable[List[NameBodyTokens]]:
        """
        Load a single data file, returning token streams.
        :param path: the path for a single data file.
        :return Iterable of lists of (name, [body])
        """
        with open(path, 'rb') as f:
            graph = Graph()
            graph.ParseFromString(f.read())
            feature_extractor = GraphFeatureExtractor(graph,
                                                      remove_override_methods=True,
                                                      min_line_of_codes=self.config['min_line_of_codes'])
            yield feature_extractor.retrieve_methods_content()
