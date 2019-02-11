from collections import defaultdict
from typing import List, Dict

from data.graph_pb2 import FeatureNode, Graph


# TODO add a method to separate signature from body
class GraphFeatureExtractor(object):
    def __init__(self, graph: Graph, remove_short_methods: bool, remove_override_methods: bool):
        self.graph = graph
        self.edges_map = self.edge_list_to_map()
        self.tokens_to_content_map = self.map_tokens_id_to_content()
        self.remove_empty_methods = remove_short_methods
        self.remove_override_methods = remove_override_methods

    def retrieve_methods_content(self) -> List[List[str]]:
        """ Retrieve the content of every method including signature and body"""
        tokens_list = []

        method_nodes = self.find_all_method_nodes()
        for method_node in method_nodes:
            method_token_list = self.extract_body_and_signature(method_node)
            if self.remove_override_methods:
                # don't add tokens that have override in them
                if 'monkeys_at' == method_token_list[0] and 'override' == method_token_list[1]:
                    continue
            if self.remove_empty_methods and len(method_token_list) < 10:
                # don't add abstracts, short and no body methods.
                continue

            tokens_list.append(method_token_list)
        return tokens_list

    def find_all_method_nodes(self) -> List[FeatureNode]:
        """ Return list of all nodes that are method nodes. """
        return list(filter(lambda n: n.contents == "METHOD", self.graph.node))

    def edge_list_to_map(self) -> Dict[str, List[str]]:
        """ Returns mapping of each parent -> all children"""
        d = defaultdict(list)

        source_dest_list = list(map(lambda edge: (edge.sourceId, edge.destinationId), self.graph.edge))
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

    def _dfs(self, node_id: str, out: List[(str, int)]):
        """ Traverse the graph to the end, keeping track of the content and node's ID """
        leaf_children = self.edges_map[node_id]
        for child_id in leaf_children:
            if child_id in self.tokens_to_content_map:  # End node has content associated with it
                token_content = self.tokens_to_content_map[child_id]
                out.append((token_content, child_id))
            else:
                self._dfs(child_id, out)
