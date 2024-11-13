from typing import List, Tuple, Dict, Union
from .ExtractModel import HCX_003_MetaDataExecutor
import re
import raptor.config.api_config as api_config

class FinQATokenizer:
    """
    다국어(한글/영어) FinQA 데이터셋을 위한 커스텀 토크나이저
    - byte 단위로 청킹하여 한글/영어 혼용 텍스트를 안전하게 처리
    - 문장 단위로 분할하여 의미 단위 보존
    """
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size  # byte 단위
        self.row_template = "{} 항목의 {} 값은 {} 입니다;"  # 한글 템플릿으로 변경
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
        if 'table_ori' in json_data and json_data['table_ori']:
            table_text = self._table_to_text(json_data['table_ori'])
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
    
    def _format_number(self, value: str) -> str:
        """숫자 포맷팅 처리"""
        try:
            num = float(value.replace(",", ""))
            if num.is_integer():
                return format(int(num), ",")
            return format(num, ",.2f")
        except:
            return value

    def _table_to_text(self, table: List[List[str]]) -> List[str]:
        """
        테이블을 자연어 문장 리스트로 변환
        - 헤더를 기준으로 테이블 구조 설명
        - 각 행을 하나의 청크로 처리
        """
        if not table or not table[0]:
            return []
            
        # 헤더 처리
        headers = table[0]
        
        # 빈 헤더 처리
        for i in range(len(headers)):
            if not headers[i].strip():
                headers[i] = f"열_{i+1}"
        
        # 테이블 설명 템플릿 수정
        self.row_template = "이 테이블의 {}에서 {} 항목의 값은 {} 입니다;"
                
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
            
            row_sentences = []
            
            # 행이 헤더보다 짧은 경우 패딩
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            
            # 행의 각 셀을 문장으로 변환
            for col_idx, value in enumerate(row[:len(headers)]):
                # 빈 값이나 구분선("-") 스킵
                if not value.strip() or value.strip().replace("-", "") == "":
                    continue
                # 이미 행 설명에 사용된 값은 스킵
                if value.strip() == row_description:
                    continue
                    
                # 특수문자 처리
                value = value.replace("\n", " ")
                value = re.sub(r'\s+', ' ', value).strip()
                
                # 숫자 포맷팅 처리
                if value.replace(".", "").replace("-", "").replace(",", "").isdigit():
                    value = self._format_number(value)
                
                # 개별 셀에 대한 문장 생성
                if headers[col_idx] and value:
                    sentence = self.row_template.format(
                        row_description,  # 행 설명
                        headers[col_idx].strip(),  # 헤더
                        value  # 값
                    )
                    row_sentences.append(sentence)
            
            # 한 행의 모든 문장을 하나로 결합
            if row_sentences:
                row_text = " ".join(row_sentences)
                # 결합된 행이 너무 길면 분할
                if len(row_text) > self.chunk_size:
                    chunks = self._split_long_sentence(row_text, self.chunk_size)
                    row_texts.extend(chunks)
                else:
                    row_texts.append(row_text)
        
        return row_texts

    def _split_long_sentence(self, sentence: str, max_length: int = 1000) -> List[str]:
        """
        긴 문장을 더 작은 청크로 분할
        """
        if len(sentence) <= max_length:
            return [sentence]
        
        chunks = []
        words = sentence.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1  # 공백 포함
            if current_length + word_len > max_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def encode(self, json_data: Union[List[dict], dict]) -> List[Tuple[Dict, str]]:
        """
        FinQA JSON 데이터를 (메타데이터, 토큰화된 문장) 튜플 리스트로 변환
        - 테이블의 각 행을 개별 청크로 처리
        """
        # 단일 딕셔너리인 경우 리스트로 변환
        if isinstance(json_data, dict):
            json_data = [json_data]
        
        result = []
        metadata = None

        # 메타데이터 처리 (이전과 동일)
        combined_text = ""
        for doc in json_data:
            if "metadata" in doc:
                metadata = doc["metadata"]
                break
            else:
                preprocessed_text = self._preprocess_text(doc)
                if preprocessed_text:
                    combined_text += preprocessed_text + "\n\n"
        
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
            # Pre-text 청킹 및 처리 (이전과 동일)
            if 'pre_text' in document and document['pre_text']:
                pre_text_chunks = self._chunk_text(
                    [text for text in document['pre_text'] if text.strip()]
                )
                for chunk in pre_text_chunks:
                    chunk_metadata = {**metadata, "section": "pre_text"}
                    result.append((chunk_metadata, chunk))
            
            # 표 처리 - 각 행을 독립적인 청크로 처리
            if 'table_ori' in document and document['table_ori']:
                table_sentences = self._table_to_text(document['table_ori'])
                for row_sentence in table_sentences:
                    if row_sentence.strip():  # 빈 문장 제외
                        chunk_metadata = {**metadata, "section": "table"}
                        result.append((chunk_metadata, row_sentence))
            
            # Post-text 청킹 및 처리 (이전과 동일)
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