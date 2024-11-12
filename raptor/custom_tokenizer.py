from typing import List, Dict, Union

class CommaTokenizer:
    """
    우선은 한글 전용, 영어로 쓸거면 utils/split_text method 변경 필요
    """
    def encode(self, text: str) -> List[str]:
        if text.startswith("Page"):
            return [text]
        chunks = [chunk.strip() for chunk in text.split(',')]
        return [chunk for chunk in chunks if chunk]
    
    def decode(self, tokens: List[str]) -> str:
        return ', '.join(tokens)
    
    def get_text_without_page_num(self, text: str) -> str:
        if text.startswith("Page"):
            return text.split(": ", 1)[1]
        return text
    
    def __str__(self):
        return "CommaTokenizer"


class FinQATokenizer:
    def _format_table(self, table: List[List]) -> str:
        """테이블 전체를 하나의 문자열로 변환"""
        rows = []
        for row in table:
            # 각 행의 셀들을 문자열로 변환
            row_cells = []
            for cell in row:
                if isinstance(cell, (str, int, float)):
                    row_cells.append(str(cell))
                elif isinstance(cell, list):
                    row_cells.append(' '.join(str(item) for item in cell if item))
            # 행을 | 로 구분된 문자열로 만들기
            if row_cells:
                rows.append(" | ".join(row_cells))
        
        # 전체 테이블을 줄바꿈으로 구분된 하나의 문자열로 만들기
        return "\n".join(rows)

    def _format_text_section(self, texts: List) -> str:
        """pre_text나 post_text 섹션을 하나의 문자열로 변환"""
        formatted_texts = []
        for item in texts:
            if isinstance(item, str) and item.strip():
                formatted_texts.append(item.strip())
            elif isinstance(item, list):
                # 리스트인 경우 각 요소를 평탄화
                for subitem in item:
                    if isinstance(subitem, str) and subitem.strip():
                        formatted_texts.append(subitem.strip())
        return " ".join(formatted_texts)

    def encode(self, data: Dict) -> Dict:
        """
        FinQA 문서를 인코딩하고 청크로 분할합니다.
        """
        doc_id = data.get('id', '')
        
        # pre_text 처리 - 전체를 하나의 청크로
        pre_text_chunks = []
        pre_text = data.get('pre_text', [])
        if isinstance(pre_text, list) and pre_text:
            formatted_pre_text = self._format_text_section(pre_text)
            if formatted_pre_text.strip():
                pre_text_chunks.append(formatted_pre_text)
        
        # table 처리 - 전체 테이블을 하나의 청크로
        table_chunks = []
        table = data.get('table', [])
        if isinstance(table, list) and table:
            formatted_table = self._format_table(table)
            if formatted_table.strip():
                table_chunks.append(formatted_table)
        
        # post_text 처리 - 전체를 하나의 청크로
        post_text_chunks = []
        post_text = data.get('post_text', [])
        if isinstance(post_text, list) and post_text:
            formatted_post_text = self._format_text_section(post_text)
            if formatted_post_text.strip():
                post_text_chunks.append(formatted_post_text)
        
        # 전체 청크 리스트 생성
        all_chunks = pre_text_chunks + table_chunks + post_text_chunks
        
        # 전체 텍스트 생성 (필요한 경우를 위해)
        full_text = "\n\n".join(all_chunks) if all_chunks else ""
        
        return {
            'id': doc_id,
            'pre_text_chunks': pre_text_chunks,
            'table_chunks': table_chunks,
            'post_text_chunks': post_text_chunks,
            'all_chunks': all_chunks,
            'full_text': full_text
        }
    
# 토크나이저 초기화
class SafeFinQATokenizer(FinQATokenizer):
    def _format_table(self, table: List[List]) -> str:
        """테이블 전체를 하나의 문자열로 변환"""
        rows = []
        for row in table:
            # 각 행의 셀들을 문자열로 변환
            row_cells = []
            for cell in row:
                if isinstance(cell, (str, int, float)):
                    row_cells.append(str(cell))
                elif isinstance(cell, list):
                    row_cells.append(' '.join(str(item) for item in cell if item))
            # 행을 | 로 구분된 문자열로 만들기
            if row_cells:
                rows.append(" | ".join(row_cells))
        
        # 전체 테이블을 줄바꿈으로 구분된 하나의 문자열로 만들기
        return "\n".join(rows)

    def _format_text_section(self, texts: List) -> str:
        """pre_text나 post_text 섹션을 하나의 문자열로 변환"""
        formatted_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                formatted_texts.append(text.strip())
            elif isinstance(text, list):
                for subitem in text:
                    if isinstance(subitem, str) and subitem.strip():
                        formatted_texts.append(subitem.strip())
        return " ".join(formatted_texts)

    def encode(self, data: Dict) -> Dict:
        """
        FinQA 문서를 인코딩하고 청크로 분할합니다.
        """
        doc_id = data.get('id', '')
        
        # pre_text 처리 - 전체를 하나의 청크로
        pre_text_chunks = []
        pre_text = data.get('pre_text', [])
        if isinstance(pre_text, list) and pre_text:
            formatted_pre_text = self._format_text_section(pre_text)
            if formatted_pre_text.strip():
                pre_text_chunks.append(formatted_pre_text)
        
        # table 처리 - 전체 테이블을 하나의 청크로
        table_chunks = []
        table = data.get('table', [])
        if isinstance(table, list) and table:
            formatted_table = self._format_table(table)
            if formatted_table.strip():
                table_chunks.append(formatted_table)
        
        # post_text 처리 - 전체를 하나의 청크로
        post_text_chunks = []
        post_text = data.get('post_text', [])
        if isinstance(post_text, list) and post_text:
            formatted_post_text = self._format_text_section(post_text)
            if formatted_post_text.strip():
                post_text_chunks.append(formatted_post_text)
        
        # 전체 청크 리스트 생성
        all_chunks = pre_text_chunks + table_chunks + post_text_chunks
        
        return {
            'id': doc_id,
            'pre_text_chunks': pre_text_chunks,
            'table_chunks': table_chunks,
            'post_text_chunks': post_text_chunks,
            'all_chunks': all_chunks
        }