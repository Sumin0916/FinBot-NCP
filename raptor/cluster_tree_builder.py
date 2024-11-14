import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering, FinRAG_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params
        self.summarization_length = 100
        
        if clustering_algorithm == RAPTOR_Clustering:
            self.layer_summarization_lengths = 100  # RAPTOR는 모든 레이어에서 100 고정
        if clustering_algorithm == FinRAG_Clustering:  # FinRAG_Clustering인 경우
            self.layer_summarization_lengths = {
                0: 300,  # Layer 0 -> 1
                1: 200,  # Layer 1 -> 2
                2: 100,  # Layer 2 -> 3
                3: 1000, # Layer 3 -> 4
            }

    def get_summarization_length(self, layer: int) -> int:
        """각 레이어의 요약 길이를 반환"""
        return self.layer_summarization_lengths.get(layer, 300)  # 기본값 300

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        if self.clustering_algorithm != RAPTOR_Clustering:
            cluster_tree_summary += f"\nLayer Summarization Lengths: {self.layer_summarization_lengths}"
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        # 부모 클래스 초기화 먼저 수행
        super().__init__(config)
        
        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        
        # ClusterTreeBuilder 특정 속성 설정
        self.config = config  # config 명시적 설정
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm()  # 인스턴스 생성
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def get_summarization_length(self, layer: int) -> int:
        """해당 레이어의 요약 길이를 반환"""
        return self.config.get_summarization_length(layer)

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster, new_level_nodes, next_node_index, layer, lock
        ):
            node_texts = get_text(cluster)

            summarization_length = self.get_summarization_length(layer)  # self.config 대신 메서드 사용

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Layer {layer} - Node Texts Length: {len(self.tokenizer.encode(node_texts))}, "
                f"Summarized Text Length: {len(self.tokenizer.encode(summarized_text))} "
                f"(Target Length: {summarization_length})"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                layer=layer,  # 현재 레이어 정보 전달
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.config.get_summarization_length(layer)
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            layer,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        layer,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
