import logging
import os
from typing import Dict, List, Set, Union, Tuple

import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer

    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            Tokenizer: {tokenizer}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Context Embedding Model: {context_embedding_model}
            Embedding Model: {embedding_model}
            Num Layers: {num_layers}
            Start Layer: {start_layer}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
        )
        return config_log


class TreeRetriever(BaseRetriever):

    def __init__(self, config, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        self.tree = tree
        self.num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        logging.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_model.create_embedding(text)

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        embeddings = get_embeddings(node_list, self.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)

        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:

            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, 
        current_nodes: List[Node], 
        query: str, 
        num_layers: int
    ) -> Tuple[List[Dict], str]:
        """
        Retrieves the most relevant information from the tree based on the query.
        """
        if not current_nodes:
            return [], ""
            
        query_embedding = self.create_embedding(query)
        selected_node_info = []

        # 노드 리스트가 정수인 경우 Node 객체로 변환
        node_list = []
        for node in current_nodes:
            if isinstance(node, int):
                if node in self.tree.all_nodes:
                    node_list.append(self.tree.all_nodes[node])
            else:
                node_list.append(node)

        for layer in range(num_layers):
            if not node_list:
                break
                
            embeddings = get_embeddings(node_list, self.context_embedding_model)
            
            if len(embeddings) == 0:
                continue
                
            distances = distances_from_embeddings(query_embedding, embeddings)
            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.threshold
                ]
            else:  # "top_k" mode
                best_indices = indices[:min(self.top_k, len(indices))]

            # 노드 정보 수집
            for idx in best_indices:
                try:
                    node = node_list[idx]
                    if isinstance(node, int):
                        if node in self.tree.all_nodes:
                            node = self.tree.all_nodes[node]
                        else:
                            continue

                    # 노드 정보 생성
                    node_info = {
                        "node_id": node.index if hasattr(node, 'index') else str(id(node)),
                        "layer": self.tree_node_index_to_layer.get(
                            node.index if hasattr(node, 'index') else str(id(node)), 
                            -1
                        ),
                        "text": node.text if hasattr(node, 'text') else str(node),
                        "metadata": node.metadata if hasattr(node, 'metadata') else {},
                        "similarity_score": float(distances[idx]),
                        "children": []
                    }

                    # 자식 노드 처리
                    if hasattr(node, 'children') and node.children:
                        child_indices = []
                        for child in node.children:
                            if isinstance(child, Node):
                                child_indices.append(child.index)
                            elif isinstance(child, int):
                                child_indices.append(child)
                        node_info["children"] = child_indices

                    selected_node_info.append(node_info)
                    
                except Exception as e:
                    logging.error(f"Error processing node {idx}: {str(e)}")
                    continue

            # 다음 레이어의 자식 노드 처리
            if layer != num_layers - 1:
                next_layer_nodes = []
                for index in best_indices:
                    current_node = node_list[index]
                    if hasattr(current_node, 'children') and current_node.children:
                        for child in current_node.children:
                            if isinstance(child, int):
                                if child in self.tree.all_nodes:
                                    next_layer_nodes.append(self.tree.all_nodes[child])
                            else:
                                next_layer_nodes.append(child)
                
                node_list = list(dict.fromkeys(next_layer_nodes))  # 중복 제거

        # 컨텍스트 생성
        context = "\n".join([
            info["text"] 
            for info in selected_node_info 
            if info.get("text")
        ])
        
        return selected_node_info, context

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_metadata: bool = True,
        format_output: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]], str]:
        """
        Enhanced retrieve method with metadata support and error handling.
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        # Validate and set defaults
        start_layer = min(self.start_layer if start_layer is None else start_layer, 
                         self.tree.num_layers)
        num_layers = min(self.num_layers if num_layers is None else num_layers,
                        start_layer + 1)

        try:
            if collapse_tree:
                selected_nodes, context = self.retrieve_information_collapse_tree(
                    query, top_k, max_tokens
                )
                # Convert to consistent format
                node_info = [
                    {
                        "node_id": node.index,
                        "layer": self.tree_node_index_to_layer.get(node.index, -1),
                        "text": node.text,
                        "metadata": node.metadata if hasattr(node, "metadata") else {},
                    }
                    for node in selected_nodes
                ]
            else:
                layer_nodes = self.tree.layer_to_nodes.get(start_layer, [])
                if not layer_nodes:
                    logging.warning(f"No nodes found in layer {start_layer}")
                    return ("", []) if return_metadata else ""
                    
                node_info, context = self.retrieve_information(
                    layer_nodes, query, num_layers
                )

            if format_output:
                # Create a formatted string representation with error handling
                output_parts = []
                for info in node_info:
                    try:
                        part = f"Node {info.get('node_id', 'Unknown')} (Layer {info.get('layer', 'Unknown')}):\n"
                        if info.get('metadata'):
                            part += "Metadata:\n"
                            for key, value in info['metadata'].items():
                                part += f"  {key}: {value}\n"
                        part += f"Text: {info.get('text', 'No text available')}\n"
                        if 'similarity_score' in info:
                            part += f"Similarity Score: {info['similarity_score']:.4f}\n"
                        if 'children' in info:
                            part += f"Child Nodes: {info['children']}\n"
                        part += "-" * 50 + "\n"
                        output_parts.append(part)
                    except Exception as e:
                        logging.error(f"Error formatting node info: {str(e)}")
                        continue
                return "\n".join(output_parts)

            return (context, node_info) if return_metadata else context

        except Exception as e:
            logging.error(f"Error in retrieve: {str(e)}")
            return ("", []) if return_metadata else ""