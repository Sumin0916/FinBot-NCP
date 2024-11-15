import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from collections import defaultdict

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
# Import necessary methods from other modules
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    """클러스터링 알고리즘을 위한 기본 추상 클래스"""
    
    @abstractmethod
    def perform_clustering(
        self,
        nodes: List[Node],
        embedding_model_name: str,
        layer: int = 0,
        **kwargs
    ) -> List[List[Node]]:
        """클러스터링 수행을 위한 추상 메서드"""
        pass
    
    @abstractmethod
    def get_summary_length(self, layer: int) -> int:
        """레이어별 요약 길이를 반환하는 추상 메서드"""
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    """RAPTOR 클러스터링 구현"""
    
    def __init__(self, max_length_in_cluster: int = 3500, summary_length: int = 300):
        self.max_length_in_cluster = max_length_in_cluster
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.summary_length = summary_length  # 모든 레이어에 대해 동일한 요약 길이 사용
    
    def perform_clustering(
        self,
        nodes: List[Node],
        embedding_model_name: str,
        layer: int = 0,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        **kwargs
    ) -> List[List[Node]]:
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        clusters = perform_clustering(embeddings, dim=reduction_dimension, threshold=threshold)
        
        node_clusters = []
        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]
            
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue
            
            total_length = sum(
                len(self.tokenizer.encode(node.text)) for node in cluster_nodes
            )
            
            if total_length > self.max_length_in_cluster:
                node_clusters.extend(
                    self.perform_clustering(
                        cluster_nodes,
                        embedding_model_name,
                        layer,
                        reduction_dimension,
                        threshold
                    )
                )
            else:
                node_clusters.append(cluster_nodes)
        
        return node_clusters
    
    def get_summary_length(self, layer: int) -> int:
        """모든 레이어에 대해 동일한 요약 길이 반환"""
        return self.summary_length


class FinRAG_Clustering(ClusteringAlgorithm):
    """금융 문서를 위한 메타데이터 기반 계층적 클러스터링"""
    logging.getLogger
    def __init__(self):
        self.clustering_config = {
            0: {
                'metadata_keys': ['sector', 'companyName', 'year'],
                'summary_length': 300,
                'description': 'Initial metadata'
            },
            1: {
                'metadata_keys': ['year'],  # Squash Years
                'summary_length': 200,
                'description': 'Year-level clustering'
            },
            2: {
                'metadata_keys': ['sector', 'companyName'],  # Group by sector and company
                'summary_length': 100,
                'description': 'Company-level clustering'
            },
            3: {
                'metadata_keys': ['sector'],  # Group by sector
                'summary_length': 1000,
                'description': 'Sector-level clustering'
            },
            4: {
                'metadata_keys': [],  # Finally group
                'summary_length': 1500,
                'description': 'Final-level clustering'
            }
        }

    def get_summary_length(self, layer: int) -> int:
        """각 레이어의 요약 길이를 반환"""
        return self.clustering_config[layer]['summary_length']
    
    def create_metadata_for_cluster(self, nodes: List[Node], layer: int = 0) -> Dict:
        """클러스터의 메타데이터 생성
        
        Args:
            nodes: 클러스터에 포함된 노드 리스트
            layer: 현재 레이어 번호 (기본값: 0)
        """
        if not nodes:
            return {}
            
        base_metadata = {}
        
        # 현재 레이어에서 유지해야 할 메타데이터 키 가져오기
        metadata_keys = self.clustering_config[layer]['metadata_keys']
        
        # 각 메타데이터 키에 대해 처리
        for key in metadata_keys:
            values = {node.metadata.get(key) for node in nodes}
            if len(values) == 1:  # 모든 노드가 같은 값을 가질 때
                base_metadata[key] = next(iter(values))  # 유일한 값 사용
            else:
                base_metadata[key] = 'all'  # 다른 값이 있을 경우 'all' 사용
                
        return base_metadata
    
    def perform_clustering(self, nodes: List[Node], embedding_model, layer: int = 0, **kwargs) -> List[List[Node]]:
        config = self.clustering_config[layer]
        metadata_keys = config['metadata_keys']
        description = config['description']

        logging.info(f"Layer {layer} - {description}: Clustering by {metadata_keys}")
        
        if layer >= 4:  # 마지막 레이어에 도달했을 때
            logging.info(f"Final summary layer reached")
            return [nodes]  # 모든 노드를 하나의 클러스터로

        # 메타데이터 기반 그룹핑
        groups = defaultdict(list)
        
        for node in nodes:
            if layer == 1:  # year 기반 클러스터링
                # year만으로 그룹화
                group_key = str(node.metadata.get('year', 'unknown'))
                groups[group_key].append(node)
                
            elif layer == 2:  # sector와 companyName 기반 클러스터링
                # sector와 companyName을 조합하여 그룹화
                sector = node.metadata.get('sector', 'unknown')
                company = node.metadata.get('companyName', 'unknown')
                group_key = f"{sector}_{company}"
                groups[group_key].append(node)
                
            elif layer == 3:  # sector 기반 클러스터링
                # sector만으로 그룹화
                group_key = str(node.metadata.get('sector', 'unknown'))
                groups[group_key].append(node)
                
            elif layer == 0:  # 초기 메타데이터 기반 클러스터링 (year, sector, companyName)
                # 모든 메타데이터를 조합하여 가장 상세한 그룹화
                year = node.metadata.get('year', 'unknown')
                sector = node.metadata.get('sector', 'unknown')
                company = node.metadata.get('companyName', 'unknown')
                group_key = f"{year}_{sector}_{company}"
                groups[group_key].append(node)

        logging.info(f"Created {len(groups)} clusters")
        
        # 빈 클러스터 제거
        valid_groups = [group for group in groups.values() if group]

        # 클러스터의 메타데이터 생성 및 로깅
        for group in valid_groups:
            group_meta = self.create_metadata_for_cluster(group, layer)
            logging.info(f"Group (Year={group_meta.get('year', 'all')}, "
                        f"Sector={group_meta.get('sector', 'all')}, "
                        f"Company={group_meta.get('companyName', 'all')}): "
                        f"{len(group)} nodes")
        
        return valid_groups