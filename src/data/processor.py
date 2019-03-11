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


def get_data_files_from_directory(data_dir, skip_tests=True, max_num_files=None) -> np.ndarray:
    files = iglob(os.path.join(data_dir, '**/*.{}'.format(DATA_FILE_EXTENSION)), recursive=True)

    # Skip tests and exception classes
    if skip_tests:
        files = filter(
            lambda file: not file.endswith(("Test.java.proto",
                                            "TestCase.java.proto",
                                            "Exception.java.proto",
                                            "Testing.java.proto",
                                            "Tests.java.proto",
                                            "IT.java.proto",
                                            "Interface.java.proto"
                                            )),
            files)
    if max_num_files:
        files = sorted(files)[:int(max_num_files)]
    else:
        files = list(files)
    np.random.shuffle(files)
    return np.array(files)


class Processor(object):

    def __init__(self, config: Dict[str, Any], data_files: List[str],
                 max_num_files: int = None, vocabulary: Vocabulary = None):
        """
        :param config: dictionary containing parsers configs and vocabulary size.
            DEFAULT_CONFIG = {
            'vocabulary_max_size':  the vocabulary embedding maximum size.
            'max_chunk_length': the maximum size of a token, smaller tokens will be padded to size.
            'vocabulary_count_threshold': the minimum occurrences of a token to not be considered a rare token.
            'min_line_of_codes':  minimum line of codes the method should contain to be considered in the corpus.
        }
        :param data_dir: path to data input directory
        :param max_num_files: Maximal number of files to load.
        :param vocabulary: (Optional) corpus vocabulary, if not given will build it from the input.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_num_files = max_num_files
        self.data_files = data_files
        self.corpus_methods_token = self.get_tokens_from_dir()
        if vocabulary is None:
            self.logger.info("No vocabulary found, building own vocabulary")
            vocabulary = self.load_vocabulary()
        self.vocabulary = vocabulary

    def load_vocabulary(self) -> Vocabulary:
        """ Return model vocabulary such as a vocabulary. """
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
        return token_vocab

    def get_tensorise_data(self) -> LoadedSamples:
        """ Returns a tensoirsed data representation from directory path"""
        return self.load_data_from_raw_sample_sequences(token_seq for token_seq in self.corpus_methods_token)

    def load_data_from_raw_sample_sequences(self, files_token_seqs: Iterable[List[NameBodyTokens]]) -> LoadedSamples:
        """
        Load and tensorise data from a file.
        :param files_token_seqs: Sequences of tokens per file to load samples from.
        :return The loaded data, as a dictionary mapping names to numpy arrays.
        """
        loaded_data = {'name_tokens': [], 'body_tokens': []}

        max_chunk_length = self.config['max_chunk_length']
        vocab = self.vocabulary

        for file_token_seqs in files_token_seqs:
            for (method_name, method_body) in file_token_seqs:
                loaded_data['name_tokens'].append(vocab.get_id_or_unk_multiple(method_name,
                                                                               pad_to_size=max_chunk_length))
                loaded_data['body_tokens'].append(vocab.get_id_or_unk_multiple(method_body,
                                                                               pad_to_size=max_chunk_length))

        assert len(loaded_data['body_tokens']) == len(loaded_data['name_tokens']), \
            "Loaded 'body_tokens' and 'name_tokens' lists need to be aligned and of" \
            + "the same length!"

        loaded_data['name_tokens'] = np.array(loaded_data['name_tokens'])
        loaded_data['body_tokens'] = np.array(loaded_data['body_tokens'])

        return loaded_data

    def get_tokens_from_dir(self) -> List[List[NameBodyTokens]]:
        """ Returns a list of all tokens in the data files. """
        return [methods_token for file in self.data_files for methods_token in self.load_data_file(file)]

    def load_data_file(self, path: str) -> Iterable[List[NameBodyTokens]]:
        """
        Load a single data file, returning token streams.
        :param path: the path for a single data file.
        :return Iterable of lists of (name, [body])
        """
        try:
            with open(path, 'rb') as f:
                graph = Graph()
                graph.ParseFromString(f.read())
                feature_extractor = GraphFeatureExtractor(graph,
                                                          remove_override_methods=True,
                                                          min_line_of_codes=self.config['min_line_of_codes'])
                yield feature_extractor.retrieve_methods_content()
        # TODO separate this into multiple exceptions and use it to skip tests and others files
        except Exception as e:
            # This means the method isn't traditional in the sense that it either doesn't contain body
            # (abstract method) or is an anonymous function, in both cases they are not input the model accepts,
            # so the warning can be safely ignored.
            self.logger.warning("Failed to load data from path: {}. Exception: {}".format(path, e))
