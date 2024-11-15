import logging
import re
from typing import Dict, List, Set, Union, Tuple, Any
from itertools import chain
import html
import numpy as np
import tiktoken
from scipy import spatial

from .tree_structures import Node
from .ExtractModel import HCX_003_MetaDataExecutor, CLOVASegmentationExecutor
from .config.api_config import OPENAI_EMBEDDING_MAX_TOKENS

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
    Returns:
        List[str]: A list of text chunks.
    """
    # Split the text into sentences using multiple delimiters
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    
    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue
        
        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            
            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chuk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]
            
            sub_chunk = []
            sub_length = 0
            
            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    
                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        
        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count
        
        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        if isinstance(node, tuple):
            node = node[1]
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)


def limit_tokens(text: str, max_tokens: int=OPENAI_EMBEDDING_MAX_TOKENS) -> str:
        """토큰 수 제한"""
        tokens = len(text)
        if tokens > max_tokens:
            return text[:int(max_tokens*0.8)]
        return text

def chunk_financial_data(json_data: List[List[dict]]) -> List[Tuple[Dict, str]]:
    """
    재무제표 데이터를 청크로 분할하고 메타데이터와 함께 반환
    중복되는 텍스트는 제외하여 처리
    
    Args:
        json_data: 재무제표 JSON 데이터 리스트
        
    Returns:
        List[Tuple[Dict, str]]: (메타데이터, 청크) 튜플의 리스트
    """
    
    def clean_text(text: str) -> str:
        """텍스트 정제"""
        if not isinstance(text, str):
            return ""
            
        # HTML 엔터티 디코딩
        text = html.unescape(text)
        
        # &#x 형태의 유니코드 문자 처리
        text = re.sub(r'&#x[0-9a-fA-F]+;?', '', text)
        
        # 연속된 공백, 탭, 줄바꿈을 단일 공백으로 변환
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 특수문자 제거 또는 변환
        text = re.sub(r'[\u200b\ufeff\xa0]', '', text)
        
        return text.strip()

    def process_text_list(text_list: List[str], seen_texts: Set[str], max_tokens: int = OPENAI_EMBEDDING_MAX_TOKENS) -> List[str]:
        """텍스트 리스트를 토큰 제한에 맞게 처리하고 중복 제거"""
        if not text_list:
            return []
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for text in text_list:
            cleaned_text = clean_text(text)
            if not cleaned_text or cleaned_text in seen_texts:
                continue
                
            seen_texts.add(cleaned_text)
            text_tokens = len(cleaned_text)
            
            if current_length + text_tokens > max_tokens * 0.8:
                if current_chunk:
                    combined_text = ' '.join(current_chunk)
                    limited_text = limit_tokens(combined_text, max_tokens)
                    chunks.append(limited_text)
                current_chunk = [cleaned_text]
                current_length = text_tokens
            else:
                current_chunk.append(cleaned_text)
                current_length += text_tokens
        
        if current_chunk:
            combined_text = ' '.join(current_chunk)
            limited_text = limit_tokens(combined_text, max_tokens)
            chunks.append(limited_text)
            
        return chunks

    def process_table(table: List[List[str]], seen_texts: Set[str], llm_table: Union[str, List[str]] = "") -> str:
        """
        테이블 데이터 처리 및 중복 제거
        
        Args:
            table: 원본 테이블 데이터 (2차원 리스트)
            seen_texts: 이미 처리된 텍스트 집합
            llm_table: LLM이 처리한 테이블 문자열 또는 문자열 리스트
        
        Returns:
            str: 처리된 테이블 문자열
        """
        if not table:
            return ""
        
        # LLM 처리된 테이블이 있으면 그것을 사용
        if isinstance(llm_table, list):
            llm_text = ' '.join(llm_table)
        else:
            llm_text = str(llm_table)
            
        if llm_text.strip():
            cleaned_llm_text = clean_text(llm_text)
            if cleaned_llm_text in seen_texts:
                return ""
            seen_texts.add(cleaned_llm_text)
            return limit_tokens(cleaned_llm_text, OPENAI_EMBEDDING_MAX_TOKENS)
        
        # LLM 처리된 테이블이 없으면 원본 테이블 사용
        cleaned_table = [[clean_text(str(cell)) for cell in row] for row in table]
        table_str = '\n'.join(['\t'.join(row) for row in cleaned_table])
        
        cleaned_table_str = clean_text(table_str)
        if cleaned_table_str in seen_texts:
            return ""
        seen_texts.add(cleaned_table_str)
        
        return limit_tokens(cleaned_table_str, OPENAI_EMBEDDING_MAX_TOKENS)

    results = []
    seen_texts = set()  # 중복 체크를 위한 집합
    
    for doc in json_data:
        for section in doc:
            if "metadata" not in section:
                logging.warning(f"메타데이터가 없는 섹션을 발견했습니다: {section}")
                continue
                
            metadata = section["metadata"]
            
            # pre_text 처리 (리스트로 되어있음)
            if section.get("pre_text"):
                pre_chunks = process_text_list(section["pre_text"], seen_texts)
                for chunk in pre_chunks:
                    if chunk.strip():
                        results.append(({**metadata, "section_type": "pre_text"}, chunk))
            
            # table 처리 (원본 테이블과 LLM 처리된 테이블 모두 포함)
            if section.get("table_ori"):
                table_text = process_table(
                    section["table_ori"],
                    seen_texts,
                    section.get("table_llm", "")
                )
                if table_text.strip():
                    results.append(({**metadata, "section_type": "table"}, table_text))
            
            # post_text 처리 (리스트로 되어있음)
            if section.get("post_text"):
                post_chunks = process_text_list(section["post_text"], seen_texts)
                for chunk in post_chunks:
                    if chunk.strip():
                        results.append(({**metadata, "section_type": "post_text"}, chunk))
    
    return results