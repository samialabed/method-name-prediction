import logging
import os
from collections import Counter
from glob import iglob
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np
from dpu_utils.mlutils import Vocabulary

from data.graph_feature_extractor import GraphFeatureExtractor
from data.graph_pb2 import Graph

LoadedSamples = Dict[str, np.ndarray]
DATA_FILE_EXTENSION = 'proto'


class PreProcessor(object):
    DEFAULT_CONFIG = {
        'vocabulary_max_size': 5000,
        'max_chunk_length': 50,
        'vocabulary_count_threshold': 2,
        'run_name': 'default_parser'
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
        tokens_counter = Counter(token for method_token in self.corpus_methods_token
                                 for (name, body) in method_token
                                 for token in body)
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

    def load_data_from_raw_sample_sequences(self,
                                            files_token_seqs: Iterable[List[Tuple[str, List[str]]]]) -> LoadedSamples:
        """
        Load and tensorise data from a file.
        :param files_token_seqs: Sequences of tokens per file to load samples from.
        :return The loaded data, as a dictionary mapping names to numpy arrays.
        """
        loaded_data = {
            "tokens": [],
            "tokens_lengths": [],
        }

        max_chunk_length = self.config['max_chunk_length']
        vocab = self.metadata['token_vocab']

        # TODO figure out what to do with name, tensorise the body only, what about name?
        for file_token_seqs in files_token_seqs:
            for (method_name, method_body) in file_token_seqs:
                loaded_data['tokens_lengths'].append(len(method_body))
                loaded_data['tokens'].append(vocab.get_id_or_unk_multiple(method_body,
                                                                          pad_to_size=max_chunk_length))

        # Turn into numpy arrays for easier slicing later:
        assert len(loaded_data['tokens']) == len(loaded_data['tokens_lengths']), \
            "Loaded 'tokens' and 'tokens_lengths' lists need to be aligned and of" \
            + "the same length!"
        loaded_data['tokens'] = np.array(loaded_data['tokens'])
        loaded_data['tokens_lengths'] = np.array(loaded_data['tokens_lengths'])
        return loaded_data

    def get_tokens_from_dir(self) -> List[List[Tuple[str, List[str]]]]:
        """ Returns a list of all tokens in the data files. """
        return [methods_token for file in self.data_files for methods_token in self.load_data_file(file)]

    def load_data_files_from_directory(self) -> List[str]:
        files = iglob(os.path.join(self.data_dir, '**/*.%s' % DATA_FILE_EXTENSION), recursive=True)
        if self.max_num_files:
            files = sorted(files)[:int(self.max_num_files)]
        else:
            files = list(files)
        np.random.shuffle(files)
        return files

    @staticmethod
    def load_data_file(path: str) -> Iterable[List[Tuple[str, List[str]]]]:
        """
        Load a single data file, returning token streams.
        :param path: the path for a single data file.
        :return Iterable of lists of (name, [body])
        """
        with open(path, 'rb') as f:
            graph = Graph()
            graph.ParseFromString(f.read())
            feature_extractor = GraphFeatureExtractor(graph, remove_override_methods=True)
            yield feature_extractor.retrieve_methods_content()
