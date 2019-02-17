from collections import defaultdict
from typing import List, Dict, Tuple

from data.graph_pb2 import FeatureNode, Graph


class GraphFeatureExtractor(object):
    def __init__(self, graph: Graph,
                 remove_override_methods: bool,
                 min_line_of_codes: int):
        """
        Extract features from graph_pb2.py graph.

        :param graph: a graph_pb2.py graph.
        :param remove_override_methods: remove methods with override.
        :param min_line_of_codes: minimum line of codes each method should contain. including the method signature.
        """
        self.graph = graph
        self.edges_map = self.edge_list_to_map()
        self.tokens_to_content_map = self.map_tokens_id_to_content()
        self.remove_override_methods = remove_override_methods
        self.min_line_of_codes = min_line_of_codes

    def retrieve_methods_content(self) -> List[Tuple[str, List[str]]]:
        """
        Retrieve the content of every method separting the signature and body
        :return list of tuple (method, list of each token of the method's body)

        Example return: [(method1, [method1_body_tokens]), (method2, [method2_body_tokens]),...]
        """
        methods_name_body_list = []

        method_nodes = self.find_all_method_nodes()
        for method_node in method_nodes:
            if method_node.endLineNumber - method_node.startLineNumber < self.min_line_of_codes:
                continue

            method_token_list = self.extract_body_and_signature(method_node)
            if self.remove_override_methods:
                # don't add tokens that have override in them
                if 'monkeys_at' == method_token_list[0] and 'override' == method_token_list[1]:
                    continue

            name, body = self.separate_method_name_from_body(method_token_list)
            methods_name_body_list.append((name, body))
        return methods_name_body_list

    def find_all_method_nodes(self) -> List[FeatureNode]:
        """ Return list of all nodes that are method nodes. """
        return list(filter(lambda n: n.contents == "METHOD", self.graph.node))

    def edge_list_to_map(self) -> Dict[str, List[str]]:
        """ Returns mapping of each parent -> all children"""
        d = defaultdict(list)

        source_dest_list = map(lambda edge: (edge.sourceId, edge.destinationId), self.graph.edge)
        for k, v in source_dest_list:
            d[k].append(v)

        return d

    def map_tokens_id_to_content(self) -> Dict[str, str]:
        """ Returns mapping of each node to its content """
        return {token.id: token.contents.replace(" ", "").lower() for token in
                (filter(lambda n: n.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN),
                        self.graph.node))}

    def extract_body_and_signature(self, method_node: FeatureNode) -> List[str]:
        """ Returns the signature and body of a method node, sorted in order of appearing in the corpus."""
        method_token_list_out = []
        self._dfs(method_node.id, method_token_list_out)
        # Sort results and remove the token_id from the list
        method_token_list_out = list(map(lambda token: token[0],
                                         sorted(method_token_list_out, key=lambda token: token[1])))

        return method_token_list_out

    def _dfs(self, node_id: str, out: List[Tuple[str, int]]):
        """ Traverse the graph to the end, keeping track of the content and node's ID """
        leaf_children = self.edges_map[node_id]
        for child_id in leaf_children:
            if child_id in self.tokens_to_content_map:  # End node has content associated with it
                token_content = self.tokens_to_content_map[child_id]
                out.append((token_content, child_id))
            else:
                self._dfs(child_id, out)

    @staticmethod
    def separate_method_name_from_body(method_token: List[str]) -> Tuple[str, List[str]]:
        method_name = ''
        for idx, token in enumerate(method_token):
            # the method name is the first token that comes before '('
            if idx + 1 < len(method_token) and method_token[idx + 1] == 'lparen':
                method_name = token

            if token == 'lbrace':
                # the body is everything after open brace '{' up to the very end which is '}'
                body = method_token[idx + 1: len(method_token) - 1]

                assert method_name != '', 'Method name should not be empty'
                assert len(body) != 0, 'Method body should not be empty'

                return method_name, body

        raise Exception(
            'Failed to separate the method name and body from the token.')  # Should I include the token used?
