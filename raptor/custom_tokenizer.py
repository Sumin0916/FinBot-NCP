from typing import List, Tuple, Dict, Union
import re
import raptor.config.api_config as api_config

from .utils import fin_json_to_list

class FinQATokenizer:
    """
    다국어(한글/영어) FinQA 데이터셋을 위한 커스텀 토크나이저
    - byte 단위로 청킹하여 한글/영어 혼용 텍스트를 안전하게 처리
    - 문장 단위로 분할하여 의미 단위 보존
    """
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size  # byte 단위
        self.row_template = "{} 항목의 {} 값은 {} 입니다;"  # 한글 템플릿으로 변경
    
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

    def encode(self, json_data):
        """
        FinQA JSON 데이터를 (메타데이터, 토큰화된 문장) 튜플 리스트로 변환
        - 각 문서별 메타데이터 처리
        - 필수 메타데이터(companyName, sector, year) 없으면 LLM으로 추출

        하나의 문서 -> 리스트로
        """
        return json_data

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