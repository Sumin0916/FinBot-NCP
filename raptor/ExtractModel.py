import json
import logging
from typing import Dict, Any, List
import requests
import http.client

from tenacity import retry, wait_random_exponential, stop_after_attempt
from .config.api_config import (
    CLOVA_HOST, 
    CLOVA_API_KEY, 
    CLOVA_API_GATEWAY_KEY, 
    HCX_003_CONFIG,
    HCX_SEGMENTATION_CONFIG,
)
        
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


class HCX_003_MetaDataExecutor:
    def __init__(self):
        self._host = CLOVA_HOST
        self._api_key = CLOVA_API_KEY
        self._api_key_primary_val = CLOVA_API_GATEWAY_KEY
        self.request_id = HCX_003_CONFIG['request_id']
        self.endpoint = HCX_003_CONFIG['endpoint']
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

            # metadata = {
            #     "success": True,
            #     "companyName": {
            #         "name": result["기업명"]["name"],
            #         "confidence": result["기업명"]["confidence"],
            #         "evidence": result["기업명"]["근거"]
            #     },
            #     "sector": {
            #         "sector": result["산업분야"]["sector"],
            #         "confidence": result["산업분야"]["confidence"],
            #         "evidence": result["산업분야"]["근거"]
            #     }
            # }

            metadata = {
                'success': True,
                "metadata": {
                    "companyName": result["기업명"]["name"],
                    "sector": result["산업분야"]["sector"],
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
            self.logger.info("Starting metadata extraction...")
            financial_text = chunk_text(financial_text) #청크 텍스트 중복 코드임
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
                'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'text/event-stream'
            }

            full_response = ""
            with requests.post(
                "https://" + self._host + self.endpoint,
                headers=headers,
                json=request_data,
                stream=True
            ) as r:
                self.logger.debug(f"API request status code: {r.status_code}")
                
                if r.status_code != 200:
                    self.logger.error(f"API request failed with status code: {r.status_code}")
                    return {
                        "success": False,
                        "error": f"API request failed with status code: {r.status_code}"
                    }
                
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        full_response += decoded_line + "\n"
                        
            return self._parse_response(full_response)

        except requests.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            return {
                "success": False,
                "error": f"API request failed: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during extraction: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


class CLOVASegmentationExecutor:
    def __init__(self):
        """
        CLOVA Studio API를 사용하여 텍스트 세그멘테이션을 수행하는 실행기 초기화
        """
        self._host = CLOVA_HOST
        self._api_key = CLOVA_API_KEY
        self._api_key_primary_val = CLOVA_API_GATEWAY_KEY
        self.request_id = HCX_SEGMENTATION_CONFIG['request_id']
        self.endpoint = HCX_SEGMENTATION_CONFIG['endpoint']
        self.logger = logging.getLogger(__name__)

    def _send_request(self, completion_request: Dict) -> Dict[str, Any]:
        """
        CLOVA Studio API에 요청을 보내는 내부 메서드
        
        Args:
            completion_request (Dict): API 요청 데이터
            
        Returns:
            Dict[str, Any]: API 응답 결과
        """
        try:
            headers = {
                'Content-Type': 'application/json; charset=utf-8',
                'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
                'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
                'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id
            }
            
            conn = http.client.HTTPSConnection(self._host)
            conn.request('POST', self.endpoint, json.dumps(completion_request), headers)
            
            response = conn.getresponse()
            result = json.loads(response.read().decode('utf-8'))
            conn.close()
            
            self.logger.debug(f"API Response: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise

    def execute(self, text: str, post_process: bool = False, 
                seg_cnt: int = -1, alpha: float = 0.0,
                min_size: int = 300, max_size: int = 1000) -> str:
        """
        텍스트 세그멘테이션 실행
        
        Args:
            text (str): 세그멘테이션할 텍스트
            post_process (bool): 후처리 여부
            seg_cnt (int): 세그먼트 수 (-1은 자동)
            alpha (float): 알파 파라미터
            min_size (int): 최소 세그먼트 크기
            max_size (int): 최대 세그먼트 크기
            
        Returns:
            str: 세그멘테이션 결과 또는 에러 메시지
        """
        try:
            self.logger.info("Starting text segmentation...")
            
            request_data = {
                "text": text,
                "postProcess": post_process,
                "segCnt": seg_cnt,
                "alpha": alpha,
                "postProcessMinSize": min_size,
                "postProcessMaxSize": max_size
            }
            
            result = self._send_request(request_data)
            
            if result['status']['code'] == '20000':
                self.logger.info("Segmentation completed successfully")
                return result['result']['topicSeg']
            else:
                error_msg = f"Error: {result['status']['code']} - {result['status'].get('message', 'Unknown error')}"
                self.logger.error(error_msg)
                return 'Error'
                
        except Exception as e:
            self.logger.error(f"Segmentation execution failed: {str(e)}")
            return 'Error'

    def segment_text(self, text: str) -> List[List[str]]:
        """
        텍스트 세그멘테이션의 편의 메서드
        
        Args:
            text (str): 세그멘테이션할 텍스트
            
        Returns:
            List[List[str]]: 세그멘테이션 결과
        """
        text = chunk_text(text, 120000)
        return self.execute(
            text=text,
            post_process=False,
            seg_cnt=-1,
            alpha=0.0,
            min_size=300,
            max_size=1000
        )
    