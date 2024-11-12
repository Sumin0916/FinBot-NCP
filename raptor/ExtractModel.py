import json
import logging
from typing import Dict, Any, Optional
import requests

from tenacity import retry, wait_random_exponential, stop_after_attempt
from config import (
    CLOVA_HOST, 
    CLOVA_API_KEY, 
    CLOVA_API_GATEWAY_KEY, 
    EXTRACT_CONFIG,
)
        

class MetaDataExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        # URL에 https:// 스키마가 없으면 추가
        if not host.startswith(('http://', 'https://')):
            self._host = f'https://{host}'
        else:
            self._host = host
            
        # 파라미터로 받은 값 사용
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id
        self.endpoint = EXTRACT_CONFIG['endpoint']
        self.logger = logging.getLogger(__name__)
        
        self.system_prompt = {
            "role": "system",
            "content": (
                "당신은 20년 경력의 재무분석 전문가입니다. 주어진 재무제표를 체계적으로 분석하여 "
                "다음 정보를 도출해주세요:\n\n"
                "1. 기업 식별\n"
                "- 재무제표의 계정과목 구성을 검토하여 기업명 파악\n"
                "- 특징적인 자산/부채 항목 기반 기업 추정\n"
                "- 매출 구조와 영업이익 패턴 분석\n\n"
                "2. 산업 분야 파악\n"
                "- 주요 매출원가 구성요소 분석\n"
                "- 영업용 자산의 특성 확인\n"
                "- 부채 및 자본 구조의 산업별 특징 대조\n"
                "- 수익성 지표와 업종별 평균 비교\n\n"
                "출력 형식:\n"
                "{\n"
                '  "기업명": {\n'
                '    "name": "분석된 기업명",\n'
                '    "confidence": "추정 신뢰도(상/중/하)",\n'
                '    "근거": "기업명 도출 근거"\n'
                "  },\n"
                '  "산업분야": {\n'
                '    "sector": "분석된 산업분야",\n'
                '    "confidence": "추정 신뢰도(상/중/하)",\n'
                '    "근거": "산업 도출 근거"\n'
                "  }\n"
                "}\n\n"
                "분석 시 주의사항:\n"
                "1. 모든 판단은 객관적 데이터에 근거할 것\n"
                "2. 추정의 불확실성은 명확히 표시할 것"
            )
        }

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        API 응답을 파싱하여 구조화된 데이터로 변환
        
        Args:
            response_text (str): API 응답 텍스트
            
        Returns:
            Dict[str, Any]: 파싱된 메타데이터
        """
        try:
            # JSON 객체 찾기
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.error("No JSON object found in response")
                return {
                    "success": False,
                    "error": "No JSON object found in response",
                    "raw_response": response_text
                }
            
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)

            metadata = {
                "success": True,
                "company_name": {
                    "name": result["기업명"]["name"],
                    "confidence": result["기업명"]["confidence"],
                    "evidence": result["기업명"]["근거"]
                },
                "industry": {
                    "sector": result["산업분야"]["sector"],
                    "confidence": result["산업분야"]["confidence"],
                    "evidence": result["산업분야"]["근거"]
                }
            }
            
            self.logger.info("Successfully parsed metadata")
            return metadata

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_response": response_text
            }
        except KeyError as e:
            self.logger.error(f"Missing required field in response: {str(e)}")
            return {
                "success": False,
                "error": f"Missing required field: {str(e)}",
                "raw_response": response_text
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "raw_response": response_text
            }

    def extract_metadata(self, financial_text: str) -> Dict[str, Any]:
        """
        재무제표 텍스트에서 회사명과 산업분야 추출
        
        Args:
            financial_text (str): 분석할 재무제표 텍스트
            
        Returns:
            Dict[str, Any]: 추출된 메타데이터
        """
        try:
            self.logger.info("Starting metadata extraction")
            
            messages = [
                self.system_prompt,
                {"role": "user", "content": financial_text}
            ]

            request_data = {
                'messages': messages,
                'topP': 0.6,
                'topK': 0,
                'maxTokens': 1000,
                'temperature': 0.1,
                'repeatPenalty': 2.0,
                'stopBefore': [],
                'includeAiFilters': False,
                'seed': 0
            }

            headers = {
                'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
                'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
                'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'text/event-stream'
            }

            self.logger.info(f"Sending request to: {self._host + self.endpoint}")
            self.logger.info(f"Request data: {request_data}")

            response = requests.post(
                self._host + self.endpoint,
                headers=headers,
                json=request_data,
                stream=True
            )

            self.logger.info(f"Response status code: {response.status_code}")
            self.logger.info(f"Response headers: {response.headers}")

            if response.status_code != 200:
                self.logger.error(f"API request failed with status code: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API request failed with status code: {response.status_code}"
                }

            # SSE 응답을 파싱하여 실제 컨텐츠 추출
            full_content = ""
            complete_message = ""
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    self.logger.debug(f"Received line: {decoded_line}")
                    
                    if decoded_line.startswith('data:'):
                        try:
                            # 'data:' 이후의 내용 파싱
                            json_str = decoded_line[5:].strip()
                            self.logger.debug(f"Parsing JSON: {json_str}")
                            
                            event_data = json.loads(json_str)
                            if 'message' in event_data:
                                message = event_data['message']
                                if 'content' in message:
                                    content = message['content']
                                    complete_message += content
                                    self.logger.debug(f"Accumulated message: {complete_message}")
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse event data: {e}")
                            self.logger.error(f"Problematic line: {decoded_line}")
                            continue

            self.logger.info(f"Complete message: {complete_message}")

            try:
                # 최종 메시지가 JSON 형식인지 확인
                if not complete_message.strip():
                    raise ValueError("Empty response from API")

                # JSON 문자열 찾기
                start_idx = complete_message.find('{')
                end_idx = complete_message.rfind('}') + 1
                
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON object found in response")
                    
                json_str = complete_message[start_idx:end_idx]
                self.logger.info(f"Attempting to parse JSON: {json_str}")
                
                result = json.loads(json_str)
                
                return {
                    "success": True,
                    "company_name": {
                        "name": result["기업명"]["name"],
                        "confidence": result["기업명"]["confidence"],
                        "evidence": result["기업명"]["근거"]
                    },
                    "industry": {
                        "sector": result["산업분야"]["sector"],
                        "confidence": result["산업분야"]["confidence"],
                        "evidence": result["산업분야"]["근거"]
                    }
                }
            except Exception as e:
                self.logger.error(f"Failed to parse final response: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to parse final response: {str(e)}",
                    "raw_response": complete_message
                }

        except Exception as e:
            self.logger.error(f"Unexpected error during extraction: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
        
    def validate_metadata(self, metadata: Dict) -> bool:
        """
        메타데이터의 유효성을 검사
        
        Args:
            metadata (Dict): 검사할 메타데이터
            
        Returns:
            bool: 메타데이터가 유효하면 True
        """
        required_fields = {'corp_name', 'report_date', 'rcept_no', 'sector'}
        return all(field in metadata and metadata[field] for field in required_fields)

    def process_chunk_metadata(self, chunk_text: str, existing_metadata: Optional[Dict] = None) -> Dict:
        """
        청크의 메타데이터를 처리. 기존 메타데이터가 없거나 불완전한 경우 LLM을 통해 추출
        
        Args:
            chunk_text (str): 청크 텍스트
            existing_metadata (Optional[Dict]): 기존 메타데이터
            
        Returns:
            Dict: 처리된 메타데이터
        """
        try:
            # 기존 메타데이터가 있고 유효한 경우 그대로 사용
            if existing_metadata and self.validate_metadata(existing_metadata):
                self.logger.info("Using existing valid metadata")
                return existing_metadata

            # 메타데이터가 없거나 불완전한 경우 LLM으로 추출
            self.logger.info("Extracting metadata using LLM")
            extracted_data = self.extract_metadata(chunk_text)
            
            if not extracted_data.get('success'):
                self.logger.warning("Failed to extract metadata using LLM")
                return {
                    'corp_name': 'Unknown',
                    'report_date': 'Unknown',
                    'rcept_no': 'Unknown',
                    'sector': 'Unknown'
                }

            # LLM 추출 결과를 표준 형식으로 변환
            processed_metadata = {
                'corp_name': extracted_data['company_name']['name'],
                'sector': extracted_data['industry']['sector'],
                'report_date': 'Unknown',  # LLM이 추출하지 못하는 경우
                'rcept_no': 'Unknown'      # LLM이 추출하지 못하는 경우
            }

            # 기존 메타데이터에서 누락된 필드만 업데이트
            if existing_metadata:
                for key in processed_metadata:
                    if key in existing_metadata and existing_metadata[key]:
                        processed_metadata[key] = existing_metadata[key]

            return processed_metadata

        except Exception as e:
            self.logger.error(f"Error processing chunk metadata: {str(e)}")
            return {
                'corp_name': 'Error',
                'report_date': 'Error',
                'rcept_no': 'Error',
                'sector': 'Error'
            }


if __name__ == '__main__':
    # 테스트 데이터
    financial_data = """
    [재무상태표]
    자산총계: 1,000,000,000
    부채총계: 400,000,000
    자본총계: 600,000,000
    
    [손익계산서]
    매출액: 800,000,000
    영업이익: 100,000,000
    당기순이익: 80,000,000
    """
    
    MetaDataExecutor().extract_metadata(financial_data)