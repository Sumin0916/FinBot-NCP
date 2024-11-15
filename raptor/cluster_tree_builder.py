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
        if isinstance(self.layer_summarization_lengths, dict):
            return self.layer_summarization_lengths.get(layer, 300)  # 기본값 300
        return self.layer_summarization_lengths  # RAPTOR의 경우 고정값 100 반환

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
    def __init__(self, config: ClusterTreeConfig) -> None:
        super().__init__(config)
        
        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        
        self.config = config
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm()
        self.clustering_params = config.clustering_params

        logging.info(f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}")

    def check_termination_condition(self, node_list: List[Node], clusters: List[List[Node]], layer: int) -> bool:
        """클러스터링 알고리즘별 종료 조건 체크"""
        if isinstance(self.clustering_algorithm, RAPTOR_Clustering):
            if len(node_list) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(f"Tree construction stopped at layer {layer} (not enough nodes)")
                return True
        elif isinstance(self.clustering_algorithm, FinRAG_Clustering):
            if layer >= 4:  # 레이어 3까지 완료한 후 종료
                self.num_layers = layer + 1
                logging.info(f"Tree construction completed after layer 3")
                return True
        return False

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info(f"Using {self.clustering_algorithm.__class__.__name__}")
        
        next_node_index = len(all_tree_nodes)
        
        def process_cluster(
            self,
            cluster: List[Node],
            new_level_nodes: Dict[int, Node],
            next_node_index: int,
            layer: int,
            lock: Lock
        ) -> None:
            node_texts = get_text(cluster)
            summarization_length = self.config.get_summarization_length(layer)
            
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )
            
            logging.info(
                f"Layer {layer} - Nodes: {len(cluster)}, "
                f"Text Length: {len(self.tokenizer.encode(node_texts))}, "
                f"Summary Length: {len(self.tokenizer.encode(summarized_text))}, "
                f"Target Length: {summarization_length}"
            )
            
            # 클러스터의 메타데이터 생성
            cluster_metadata = {}  # 기본값 설정
            if isinstance(self.clustering_algorithm, RAPTOR_Clustering):
                cluster_metadata = self.clustering_algorithm.create_metadata_for_cluster(cluster)
            elif isinstance(self.clustering_algorithm, FinRAG_Clustering):
                cluster_metadata = self.clustering_algorithm.create_metadata_for_cluster(cluster, layer)  # layer 인자 추가
            
            __, new_parent_node = self.create_node(
                next_node_index,
                summarized_text,
                {node.index for node in cluster},
                metadata=cluster_metadata  # 생성된 메타데이터 전달
            )
            
            with lock:
                new_level_nodes[next_node_index] = new_parent_node
    
            return
        
        for layer in range(self.num_layers):
            new_level_nodes = {}
            node_list_current_layer = get_node_list(current_level_nodes)
            
            # 클러스터링 수행
            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                layer=layer,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params
            )
            
            # 알고리즘별 종료 조건 체크
            if self.check_termination_condition(node_list_current_layer, clusters, layer):
                break
            
            # 클러스터 처리
            lock = Lock()
            if use_multithreading:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for cluster in clusters:
                        future = executor.submit(
                            process_cluster,
                            self,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            layer,
                            lock
                        )
                        futures.append(future)
                        next_node_index += 1
                    for future in futures:
                        future.result()  # 모든 쓰레드 완료 대기
            else:
                for cluster in clusters:
                    process_cluster(
                        self,
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        layer,
                        lock
                    )
                    next_node_index += 1
            
            # 레이어 정보 업데이트
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)
            
            # 트리 갱신
            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )
        
        return current_level_nodes