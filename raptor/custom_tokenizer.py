from typing import List, Tuple, Dict, Union
from .ExtractModel import HCX_003_MetaDataExecutor
import re
import raptor.config as config

class FinQATokenizer:
    """
    다국어(한글/영어) FinQA 데이터셋을 위한 커스텀 토크나이저
    - byte 단위로 청킹하여 한글/영어 혼용 텍스트를 안전하게 처리
    - 문장 단위로 분할하여 의미 단위 보존
    """
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size  # byte 단위
        self.row_template = "the {} of {} is {};"
        self.metadata_executor = HCX_003_MetaDataExecutor()
    
    def _preprocess_text(self, json_data: dict) -> str:
        """
        pre_text, table, post_text를 하나의 문서로 통합하고 전처리
        Dict -> Str
        """
        document_parts = []
        
        # Pre-text 처리
        if 'pre_text' in json_data and json_data['pre_text']:
            pre_text = ' '.join([text.strip() for text in json_data['pre_text'] if text.strip()])
            if pre_text:
                document_parts.append(pre_text)
        
        # Table 처리
        if 'table' in json_data and json_data['table']:
            table_text = self._table_to_text(json_data['table'])
            if table_text:
                document_parts.append(table_text)
        
        # Post-text 처리
        if 'post_text' in json_data and json_data['post_text']:
            post_text = ' '.join([text.strip() for text in json_data['post_text'] if text.strip()])
            if post_text:
                document_parts.append(post_text)
        
        # 전체 문서 통합
        full_document = '\n\n'.join(document_parts)
        
        # 기본적인 텍스트 정제
        cleaned_text = self._clean_text(full_document)
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
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

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분리
        한글/영어 모두 고려하여 문장 경계 탐지
        """
        # 한글/영어 문장 구분자 패턴
        pattern = r'(?<=[.!?։。]\s)|(?<=[.!?։。])|(?<=\n)|(?<=\r\n)'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_byte_length(self, text: str) -> int:
        """텍스트의 byte 길이 반환"""
        return len(text.encode('utf-8'))

    def _chunk_text(self, texts: List[str]) -> List[str]:
        """
        텍스트를 byte 크기 기준으로 청킹
        문장 단위를 보존하면서 분할
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for text in texts:
            # 각 텍스트를 문장 단위로 분리
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                sentence_bytes = self._get_byte_length(sentence)
                
                # 한 문장이 chunk_size보다 큰 경우
                if sentence_bytes > self.chunk_size:
                    # 현재까지의 청크 추가
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # 긴 문장을 강제로 분할
                    words = sentence.split()
                    temp_chunk = []
                    temp_length = 0
                    
                    for word in words:
                        word_bytes = self._get_byte_length(word + " ")
                        if temp_length + word_bytes > self.chunk_size:
                            if temp_chunk:
                                chunks.append(" ".join(temp_chunk))
                            temp_chunk = [word]
                            temp_length = word_bytes
                        else:
                            temp_chunk.append(word)
                            temp_length += word_bytes
                    
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    
                # 일반적인 경우
                elif current_length + sentence_bytes + 1 > self.chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_bytes
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_bytes + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _extract_metadata(self, json_data: dict) -> Dict:
        """문서에서 메타데이터 추출"""
        metadata = {
            "filename": json_data.get("filename", ""),
            "id": json_data.get("id", ""),
            "qa": json_data.get("qa", {})
        }
        return metadata

    def _table_to_text(self, table: List[List[str]]) -> str:
        """
        표를 문자열로 변환
        """
        if not table or len(table) < 2:
            return ""
            
        header = table[0]
        rows = []
        
        for row in table[1:]:
            row_parts = []
            for col_idx, cell in enumerate(row):
                if str(cell).strip() and str(header[col_idx]).strip():
                    text = self.row_template.format(
                        header[col_idx].strip(),
                        row[0].strip(),
                        str(cell).strip()
                    )
                    row_parts.append(text)
            if row_parts:
                rows.append(" ".join(row_parts))
                
        return " ".join(rows)

    def encode(self, json_data: Union[List[dict], dict]) -> List[Tuple[Dict, str]]:
        """
        FinQA JSON 데이터를 (메타데이터, 토큰화된 문장) 튜플 리스트로 변환
        
        Args:
            json_data: FinQA 형식의 JSON 데이터 (리스트 또는 딕셔너리)
            
        Returns:
            List[Tuple[Dict, str]]: (메타데이터, 토큰화된 문장) 튜플의 리스트
        """
        # 단일 딕셔너리인 경우 리스트로 변환
        if isinstance(json_data, dict):
            json_data = [json_data]
        
        result = []
        metadata = None
    
        # 리스트 내 모든 문서를 통합하여 메타데이터 추출 시도
        combined_text = ""
        for doc in json_data:
            if "metadata" in doc:
                metadata = doc["metadata"]
                break
            else:
                preprocessed_text = self._preprocess_text(doc)
                if preprocessed_text:
                    combined_text += preprocessed_text + "\n\n"
        
        # 메타데이터를 찾지 못한 경우, 통합된 텍스트에서 추출 시도
        if metadata is None and combined_text:
            metadata_result = self.metadata_executor.extract_metadata(combined_text)
            if metadata_result.get('success'):
                metadata = metadata_result
            else:
                metadata = {
                    "companyName": "UNK",
                    "ticker": "UNK",
                    "sector": "UNK",
                    "year": "UNK",
                    "issue_d": "UNK",
                    "quarter": "UNK",
                    "market": "UNK",
                    "country": "UNK",
                }
            
        for document in json_data:
            # Pre-text 청킹 및 처리
            if 'pre_text' in document and document['pre_text']:
                pre_text_chunks = self._chunk_text(
                    [text for text in document['pre_text'] if text.strip()]
                )
                for chunk in pre_text_chunks:
                    chunk_metadata = {**metadata, "section": "pre_text"}
                    result.append((chunk_metadata, chunk))
            
            # 표 청킹 및 처리
            if 'table' in document and document['table']:
                table_sentences = self._table_to_text(document['table'])
                if table_sentences:  # 빈 문자열이 아닌 경우에만 처리
                    table_chunks = self._chunk_text([table_sentences])
                    for chunk in table_chunks:
                        chunk_metadata = {**metadata, "section": "table"}
                        result.append((chunk_metadata, chunk))
            
            # Post-text 청킹 및 처리
            if 'post_text' in document and document['post_text']:
                post_text_chunks = self._chunk_text(
                    [text for text in document['post_text'] if text.strip()]
                )
                for chunk in post_text_chunks:
                    chunk_metadata = {**metadata, "section": "post_text"}
                    result.append((chunk_metadata, chunk))
        
        return result

    def decode(self, token_pairs: List[Tuple[Dict, str]]) -> List[Tuple[Dict, str]]:
        """
        토큰화된 (메타데이터, 문장) 튜플 리스트를 그대로 반환
        
        Args:
            token_pairs: (메타데이터, 토큰화된 문장) 튜플의 리스트
            
        Returns:
            입력받은 (메타데이터, 문장) 튜플 리스트
        """
        return token_pairs

    def __str__(self):
        return "FinQATokenizer"