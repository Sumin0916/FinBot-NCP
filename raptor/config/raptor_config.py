# config/raptor_config.py

from raptor import RetrievalAugmentationConfig
from raptor.custom_tokenizer import FinQATokenizer
from raptor.SummarizationModels import HCX_003_SummarizationModel
from raptor.QAModels import HCX_003_QAModel
from raptor.cluster_utils import FinRAG_Clustering

def get_default_config():
    """
    Returns default configuration for RAPTOR
    """
    return RetrievalAugmentationConfig(
        # Tree Builder 설정
        tree_builder_type="cluster",  # cluster 타입 사용
        tb_clustering_algorithm=FinRAG_Clustering,  # FinRAG Clustering 알고리즘 사용

        tb_tokenizer=FinQATokenizer(chunk_size=1024),  # 기본 chunk_size 설정
        tr_tokenizer=FinQATokenizer(chunk_size=1024),  # retriever용 토크나이저
    
        
        # 요약 모델 설정
        summarization_model=HCX_003_SummarizationModel(),
        # QA 모델 선정
        qa_model=HCX_003_QAModel(),
        
        # TreeBuilder 파라미터
        tb_metadata_extract_model="HCX-003",
        tb_max_tokens=1000,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        
        # TreeRetriever 파라미터 
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_num_layers=None,
        tr_start_layer=None
    )

def get_custom_config(
    chunk_size=1024,
    max_tokens=1000,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    summarization_length=100
):
    """
    Returns customized configuration for RAPTOR
    
    Args:
        chunk_size (int): Size of text chunks for tokenization
        max_tokens (int): Maximum tokens per node
        num_layers (int): Number of tree layers
        threshold (float): Similarity threshold
        top_k (int): Number of top results to return
        summarization_length (int): Maximum length of summaries
    """
    return RetrievalAugmentationConfig(
        tree_builder_type="cluster",
        tb_clustering_algorithm=FinRAG_Clustering,
        tb_tokenizer=FinQATokenizer(chunk_size=chunk_size),
        tr_tokenizer=FinQATokenizer(chunk_size=chunk_size),
        
        summarization_model=HCX_003_SummarizationModel(),
        qa_model=HCX_003_QAModel(),
        
        tb_metadata_extract_model="HCX-003",
        tb_max_tokens=max_tokens,
        tb_num_layers=num_layers,
        tb_threshold=threshold,
        tb_top_k=top_k,
        tb_selection_mode="top_k",
        tb_summarization_length=summarization_length,
        
        tr_threshold=threshold,
        tr_top_k=top_k,
        tr_selection_mode="top_k",
        tr_num_layers=None,
        tr_start_layer=None
    )