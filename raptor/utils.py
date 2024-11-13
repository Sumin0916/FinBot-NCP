import logging
import re
from typing import Dict, List, Set, Union, Tuple, Any

import numpy as np
import tiktoken
from scipy import spatial

from .tree_structures import Node
from .ExtractModel import HCX_003_MetaDataExecutor, CLOVASegmentationExecutor

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


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
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

def clean_text(text: str) -> str:
    """
    텍스트 정제를 위한 기본적인 전처리
    """
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 불필요한 특수문자 제거 또는 변환
    text = re.sub(r'[\u200b\ufeff\xa0]', ' ', text)
    # 줄바꿈 표준화
    text = re.sub(r'\r\n?', '\n', text)
    # 빈 줄 최소화
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def split_into_sentences(text: str) -> List[List[str]]: ## 추후 변경 가능성 높음
    """
    텍스트를 문장 단위로 분리 (클로바 문단 분리기 사용함)
    한글/영어 모두 고려하여 문장 경계 탐지
    """
    executor = CLOVASegmentationExecutor()
    return executor.segment_text(text)

def chunk_text(text: str, max_tokens: int) -> str:
    """
    텍스트를 길이 기준으로 검사하고 필요시 절삭
    
    Args:
        text (str): 처리할 텍스트
        max_tokens (int): 최대 허용 토큰 수
    
    Returns:
        str: 처리된 텍스트
    """
    logger = logging.getLogger(__name__)
    
    # 텍스트 길이의 80%를 계산
    text_length = len(text)
    threshold_length = int(text_length * 0.8)
    
    # 80% 길이가 최대 토큰 수를 초과하는지 확인
    if threshold_length > max_tokens:
        # 로깅
        logger.warning(
            f"텍스트가 최대 허용 길이를 초과합니다.\n"
            f"원본 길이: {text_length}\n"
            f"80% 길이: {threshold_length}\n"
            f"최대 토큰: {max_tokens}\n"
            f"텍스트가 {max_tokens}토큰까지 절삭됩니다."
        )
        
        # 텍스트를 max_tokens까지 잘라내기
        # 문장 중간 절단을 피하기 위해 마지막 마침표나 띄어쓰기를 찾아서 자름
        cutoff_text = text[:max_tokens]
        last_period = cutoff_text.rfind('.')
        last_space = cutoff_text.rfind(' ')
        
        # 마침표나 띄어쓰기 중 더 뒤에 있는 위치에서 자르기
        cut_position = max(last_period, last_space)
        if cut_position == -1:  # 마침표나 띄어쓰기가 없는 경우
            cut_position = max_tokens
        
        return text[:cut_position].strip()
    
    # 길이가 초과하지 않으면 원본 텍스트 반환
    return text

def table_to_text(table: List[List[str]], row_template: str = "{} 항목의 {} 값은 {} 입니다;") -> List[str]:
    """
    테이블을 자연어 문장 리스트로 변환
    """
    if not table or not table[0]:
        return []
        
    # 헤더 처리
    headers = table[0]
    
    # 빈 헤더 처리
    for i in range(len(headers)):
        if not headers[i].strip():
            headers[i] = f"열_{i+1}"
    
    # 테이블 내용을 자연어 문장으로 변환
    row_texts = []
    for row_idx in range(1, len(table)):
        row = table[row_idx]
        row_description = f"행_{row_idx}"  # 기본 행 설명
        
        # 날짜나 구분자 같은 특별한 값이 있는지 확인
        for header, value in zip(headers, row):
            if any(keyword in header.lower() for keyword in ['일자', '날짜', '기간', 'date']):
                if value and value.strip() and not value.strip().replace("-", "") == "":
                    row_description = value.strip()
                    break
            elif '구분' in header:
                if value and value.strip() and not value.strip().replace("-", "") == "":
                    row_description = value.strip()
                    break
        
        # 행이 헤더보다 짧은 경우 패딩
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        
        row_sentences = []
        for col_idx, value in enumerate(row[:len(headers)]):
            if not value.strip() or value.strip().replace("-", "") == "" or value.strip() == row_description:
                continue
            
            value = value.replace("\n", " ")
            value = re.sub(r'\s+', ' ', value).strip()
            
            if headers[col_idx] and value:
                sentence = row_template.format(
                    row_description,
                    headers[col_idx].strip(),
                    value
                )
                row_sentences.append(sentence)
        
        if row_sentences:
            row_text = " ".join(row_sentences)
            row_texts.append(row_text)
    
    return row_texts

from typing import List, Dict, Tuple, Any

from typing import List, Dict, Tuple, Any

def fin_json_to_list(json_data: List[Dict], metadata_executor: Any) -> List[Tuple[Dict, str]]:
    """
    FinQA JSON 데이터를 (메타데이터, 토큰화된 문장) 튜플 리스트로 변환
    
    Args:
        json_data: 단일 또는 복수의 FinQA 형식 문서
        metadata_executor: 메타데이터 추출을 위한 실행기
        
    Returns:
        List[Tuple[Dict, str]]: (메타데이터, 문장) 튜플의 리스트
    """
    # 단일 딕셔너리인 경우 리스트로 변환
    if isinstance(json_data, dict):
        json_data = [json_data]
    
    result = []
    all_texts = []
    
    # 모든 텍스트 수집
    for segment in json_data:
        # pre_text 처리
        if segment.get('pre_text'):
            if isinstance(segment['pre_text'], str):
                all_texts.append(segment['pre_text'])
            else:
                all_texts.extend(text for text in segment['pre_text'] if text and text.strip())
        
        # post_text 처리
        if segment.get('post_text'):
            if isinstance(segment['post_text'], str):
                all_texts.append(segment['post_text'])
            else:
                all_texts.extend(text for text in segment['post_text'] if text and text.strip())
    
    # 전체 텍스트에 대해 한 번만 메타데이터 추출 및 문장 분리
    if all_texts:
        combined_text = " ".join(all_texts)
        combined_text = clean_text(combined_text)
        metadata_result = metadata_executor.extract_metadata(combined_text)
        
        sentences = split_into_sentences(combined_text)
        for sent_list in sentences:
            for sentence in sent_list:
                result.append(({**metadata_result, "section": "text"}, sentence))
    
    # 테이블 처리
    for segment in json_data:
        if segment.get('table_llm'):
            table_chunks = tableLLM_to_chunk(segment['table_llm'])
            for chunk in table_chunks:
                result.append(({**metadata_result, "section": "table"}, chunk))
        elif segment.get('table_ori'):
            table_texts = table_to_text(segment['table_ori'])
            for text in table_texts:
                result.append(({**metadata_result, "section": "table"}, text))
    
    return result

def tableLLM_to_chunk(text: str) -> List[str]:
    """
    LLM이 생성한 테이블 설명 텍스트를 청크 리스트로 변환하는 함수
    
    Args:
        text (str): LLM이 생성한 테이블 설명 텍스트
        
    Returns:
        List[str]: [테이블 설명 + 행1 내용, 테이블 설명 + 행2 내용, ...] 형식의 청크 리스트
    
    Example:
        >>> text = "위의 표는 2021년 보고서이다. 회사명은 포스코이다. 대표이사는 김학동이다."
        >>> chunks = tableLLM_to_chunk(text)
        >>> print(chunks)
        ['위의 표는 2021년 보고서이다. 회사명은 포스코이다.',
         '위의 표는 2021년 보고서이다. 대표이사는 김학동이다.']
    """
    # 테이블 설명 추출 (첫 문장)
    sentences = re.split(r'(?<=[다])\.\s+', text)
    table_description = sentences[0] + "."
    
    # 각 문장을 청크로 변환
    chunks = []
    for sentence in sentences[1:]:
        if not sentence.strip():  # 빈 문장 제외
            continue
        # 설명과 현재 문장을 결합하여 청크 생성
        chunk = f"{table_description} {sentence}."
        chunks.append(chunk)
    
    return chunks