import logging
import re
import uuid
import requests
from typing import Dict, List, Set, Union
import numpy as np
from scipy import spatial
import tiktoken
from datetime import datetime
from config import CLOVA_API_KEY, CLOVA_API_GATEWAY_KEY

from .tree_structures import Node
from .custom_tokenizer import CommaTokenizer
from .custom_tokenizer import FinQATokenizer


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer

def flatten_text(text: Union[str, List[str]]) -> str:
    """리스트를 문자열로 평탄화합니다."""
    if isinstance(text, list):
        return ' '.join(flatten_text(t) for t in text)
    return text

def split_text(text: str, tokenizer, max_tokens: int, overlap: int = 0) -> List[str]:
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
    Args:
        text (str): The text to be split.
        tokenizer: The tokenizer (CommaTokenizer or tiktoken) to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
    Returns:
        List[str]: A list of text chunks with page numbers.
    """
    text = flatten_text(text)  # 텍스트 평탄화

    # CommaTokenizer 사용 시
    if isinstance(tokenizer, CommaTokenizer):
       base_splits = [s.strip() for s in text.split(',') if s.strip()]
       chunks = []
       
       for page_num, text_chunk in enumerate(base_splits, 1):
           # "Page N: " 형식의 prefix 길이 계산 
           page_prefix = f"Page {page_num}" #보통 4토큰
           available_tokens = max_tokens - 7
           spaces = text.count(' ')  # 공백 수
           other_chars = len(text) - spaces  # 나머지 문자 수

           text_token = other_chars * 1.7 + spaces # 한글자에 대충 2 토큰이라 생각함. (오류 발생 가능성 항상 존재)

           if text_token > available_tokens:
               # 긴 문장을 분할
               remain_text = text_chunk
               chunk_num = 1
               while remain_text:
                   # 각 분할에 대해 페이지 번호와 하위 번호 추가
                   current_prefix = f"{page_prefix}.{chunk_num}: "
                   available_space = max_tokens - 7

                   current_chunk = remain_text[:int(available_space//1.2)]
                   chunks.append(f"{current_prefix}{current_chunk}")
                   
                   remain_text = remain_text[int(available_space//1.2):]
                   chunk_num += 1
           else:
               # 문장이 충분히 짧으면 그대로 사용
               chunks.append(f"{page_prefix}: {text_chunk}")
               
       return chunks
    
    # # FinQATokenizer 사용 시
    # elif isinstance(tokenizer, FinQATokenizer):
    #     if isinstance(text, dict):
    #         # FinQA 문서 형식일 경우
    #         pre_text = ' '.join(text.get('pre_text', [])) if isinstance(text.get('pre_text'), list) else text.get('pre_text', '')
    #         post_text = ' '.join(text.get('post_text', [])) if isinstance(text.get('post_text'), list) else text.get('post_text', '')
            
    #         # 표 텍스트화
    #         table_text = ''
    #         if text.get('table'):
    #             table_rows = []
    #             for row in text['table']:
    #                 table_rows.append(' | '.join(str(cell) for cell in row))
    #             table_text = '\n'.join(table_rows)
            
    #         # 전체 텍스트를 하나의 청크로 구성
    #         full_text = f"{pre_text}\n\n{table_text}\n\n{post_text}".strip()
    #         return [full_text]  # 한 문서 전체를 하나의 청크로 반환
    #     else:
    #         # 일반 텍스트일 경우 그대로 반환
    #         return [text]
        
    # tiktoken 사용 시 (기존 로직)
    else:
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
                
                filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip()]
                sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]
                
                sub_chunk = []
                sub_length = 0
                
                for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                    if sub_length + sub_token_count > max_tokens:
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

## 추가

class HyperClovaExecutor:
    """하이퍼클로바X API 실행(메타데이터 추출을 위해)"""
    
    def __init__(
        self,
        api_key: str = CLOVA_API_KEY,
        api_key_primary: str = CLOVA_API_GATEWAY_KEY,
        host: str = "https://clovastudio.stream.ntruss.com"
    ):
        self._host = host
        self._api_key = api_key
        self._api_key_primary = api_key_primary
        self._request_id = str(uuid.uuid4())
        
    def extract_metadata(self, text: str) -> Dict:
        """재무 문서에서 메타데이터 추출"""
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        # 프롬프트 구성
        messages = [
            {
                "role": "system",
                "content": "당신은 재무제표에서 산업 섹터, 회사명, 해당 연도를 추출하는 재무 분석가입니다."
            },
            {
                "role": "user",
                "content": f"""다음 텍스트에서 산업 섹터, 회사명, 연도를 추출해주세요:

{text}

다음 형식으로 답변해주세요:
Sector: [산업 섹터]
Company: [회사명]
Year: [연도]"""
            }
        ]
        
        request_data = {
            'messages': messages,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 256,
            'temperature': 0.1,  # 정확한 추출을 위해 낮은 temperature 사용
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True
        }
        
        try:
            response = requests.post(
                f"{self._host}/testapp/v1/chat-completions/HCX-003",
                headers=headers,
                json=request_data
            )
            response.raise_for_status()
            
            # 응답 파싱
            result = response.json()
            if 'result' in result and 'message' in result['result']:
                return self._parse_response(result['result']['message']['content'])
            
        except Exception as e:
            logging.error(f"하이퍼클로바X API 호출 중 오류: {str(e)}")
            
        return {
            'sector': None,
            'company': None,
            'year': None
        }
    
    def _parse_response(self, response_text: str) -> Dict:
        """하이퍼클로바X 응답 파싱"""
        metadata = {
            'sector': None,
            'company': None,
            'year': None
        }
        
        try:
            lines = response_text.split('\n')
            for line in lines:
                if line.startswith('Sector:'):
                    metadata['sector'] = line.replace('Sector:', '').strip()
                elif line.startswith('Company:'):
                    metadata['company'] = line.replace('Company:', '').strip()
                elif line.startswith('Year:'):
                    year_str = line.replace('Year:', '').strip()
                    try:
                        year_str = ''.join(filter(str.isdigit, year_str))
                        if len(year_str) == 4:
                            metadata['year'] = int(year_str)
                    except ValueError:
                        pass
                        
        except Exception as e:
            logging.error(f"응답 파싱 중 오류: {str(e)}")
            
        return metadata

def process_finqa_document(doc: Dict, hyperclova, tokenizer: FinQATokenizer) -> Node:
    """
    FinQA 문서를 처리하고 노드를 생성합니다.
    
    Args:
        doc (Dict): FinQA 문서 데이터
        hyperclova: HyperClova API executor
        tokenizer (FinQATokenizer): 문서 토크나이저
        
    Returns:
        Node: 처리된 문서 노드
    """
    try:
        # FinQATokenizer로 문서 인코딩
        encoded_doc = tokenizer.encode(doc)
        
        if not encoded_doc['all_chunks']:
            print(f"Warning: No chunks generated for document {doc.get('id', '')}")
            return None
        
        print(f"\nProcessing document: {doc.get('id', '')}")
        print(f"Found {len(encoded_doc['pre_text_chunks'])} pre-text chunks")
        print(f"Found {len(encoded_doc['table_chunks'])} table chunks")
        print(f"Found {len(encoded_doc['post_text_chunks'])} post-text chunks")
        
        # 각 청크별로 메타데이터 추출 시도
        chunk_metadata = []
        
        # pre_text 청크에서 우선 시도
        for chunk in encoded_doc['pre_text_chunks']:
            try:
                metadata = hyperclova.extract_metadata(chunk)
                chunk_metadata.append(metadata)
                if all(metadata.values()):
                    break
            except Exception as e:
                print(f"Error processing pre-text chunk: {str(e)}")
        
        # 필요한 경우 table 청크에서 시도
        if not any(all(m.values()) for m in chunk_metadata):
            for chunk in encoded_doc['table_chunks']:
                try:
                    metadata = hyperclova.extract_metadata(chunk)
                    chunk_metadata.append(metadata)
                    if all(metadata.values()):
                        break
                except Exception as e:
                    print(f"Error processing table chunk: {str(e)}")
        
        # 필요한 경우 post_text 청크에서 시도
        if not any(all(m.values()) for m in chunk_metadata):
            for chunk in encoded_doc['post_text_chunks']:
                try:
                    metadata = hyperclova.extract_metadata(chunk)
                    chunk_metadata.append(metadata)
                    if all(metadata.values()):
                        break
                except Exception as e:
                    print(f"Error processing post-text chunk: {str(e)}")
        
        # 최적의 메타데이터 선택
        final_metadata = {'sector': None, 'company': None, 'year': None}
        if chunk_metadata:
            # 가장 많은 필드가 채워진 메타데이터 선택
            final_metadata = max(
                chunk_metadata,
                key=lambda m: sum(1 for v in m.values() if v is not None)
            )
        
        # 노드 생성
        node = Node(
            index=doc.get('id', ''),
            text=encoded_doc['full_text'],
        )
        
        # 메타데이터 추가
        node.metadata = {
            'sector': final_metadata.get('sector'),
            'company': final_metadata.get('company'),
            'year': final_metadata.get('year'),
            'filename': doc.get('filename'),
            'num_chunks': len(encoded_doc['all_chunks']),
            'chunk_types': {
                'pre_text': len(encoded_doc['pre_text_chunks']),
                'table': len(encoded_doc['table_chunks']),
                'post_text': len(encoded_doc['post_text_chunks'])
            }
        }
        
        print(f"\nExtracted metadata:")
        print(f"Sector: {final_metadata.get('sector', 'Not found')}")
        print(f"Company: {final_metadata.get('company', 'Not found')}")
        print(f"Year: {final_metadata.get('year', 'Not found')}")
        
        return node
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return None