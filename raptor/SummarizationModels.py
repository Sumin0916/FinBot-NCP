import logging
import os
from abc import ABC, abstractmethod
import http.client
import json
import requests

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from .config import (
    CLOVA_HOST, 
    CLOVA_API_KEY, 
    CLOVA_API_GATEWAY_KEY, 
    SUMMARY_CONFIG,
    HCX_003_CONFIG
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class NCPSummarizationModel(BaseSummarizationModel):
    """
    요약 전용 모델 사용 (Max Tokens를 설정할 수 없음)
    """
    def __init__(self):
        self.host = CLOVA_HOST
        self.api_key = CLOVA_API_KEY
        self.api_key_primary_val = CLOVA_API_GATEWAY_KEY
        self.request_id = SUMMARY_CONFIG['request_id']
        self.endpoint = SUMMARY_CONFIG['endpoint']

    def _send_request(self, texts, seg_min_size=300, seg_max_size=1000):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id
        }

        request_data = {
            "texts": texts if isinstance(texts, list) else [texts],
            "segMinSize": seg_min_size,
            "includeAiFilters": False,
            "autoSentenceSplitter": True,
            "segCount": -1,
            "segMaxSize": seg_max_size
        }

        conn = http.client.HTTPSConnection(self.host)
        conn.request(
            'POST',
            self.endpoint,
            json.dumps(request_data),
            headers
        )
        
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            # 입력 문자열 길이 체크
            input_length = len(context)
            if input_length > 35000:
                error_msg = f"Input text too long: {input_length} chars (Max summarize length : 35000)"
                logging.error(error_msg)
                return error_msg

            # segMaxSize를 유효 범위로 조정 (1~3000)
            seg_max_size = min(max(max_tokens, 1), 3000)
            
            response = self._send_request(
                texts=context,
                seg_max_size=seg_max_size
            )
            
            if response['status']['code'] == '20000':
                result_text = response['result']['text']
                input_tokens = response['result'].get('inputTokens', 0)  # inputTokens가 있다면 가져오기
                
                # 입력/출력 길이 로깅
                logging.info(
                    f"HyperCLOVA X Summarization - Input length: {input_length} chars, "
                    f"Input tokens: {input_tokens}, "
                    f"Output lengtdh: {len(result_text)} chars"
                )
                
                return result_text
            else:
                logging.error(f"Error in HyperCLOVA X API: {response}")
                return f"Error: {response['status']['message']}"
                
        except Exception as e:
            logging.error(f"Exception in HyperCLOVA X summarization: {e}")
            return str(e)
        

class HCX_003_SummarizationModel(BaseSummarizationModel):
    """
    HyperCLOVA X 003 모델을 사용한 요약 클래스
    Streaming API 지원 및 max_tokens 설정 가능
    """
    def __init__(self):
        self.host = CLOVA_HOST
        self.api_key = CLOVA_API_KEY
        self.api_key_primary_val = CLOVA_API_GATEWAY_KEY
        self.request_id = HCX_003_CONFIG['request_id']
        self.endpoint = HCX_003_CONFIG['endpoint']
        self.logger = logging.getLogger(__name__)

    def _prepare_messages(self, context):
        """프롬프트 메시지 준비"""
        return [
            {
            "role": "system",
            "content": """당신은 20년 경력의 재무분석 전문가입니다. 주어진 재무제표를 다음 구조로 체계적으로 분석하여 요약해주세요:

            1. 핵심 요약 (Executive Summary)
            - 가장 중요한 3-4개의 핵심 포인트
            - 전체 맥락에서 특별히 주목해야 할 변화나 트렌드

            2. 주요 수치 분석 (Key Metrics)
            - 매출/이익 지표 
            - 재무상태 지표
            - 주요 재무비율
            - 전년 대비 주요 변동사항

            3. 세부 인사이트 (Detailed Insights)
            - 수익성 관련 주요 발견점
            - 재무안정성 관련 중요 사항
            - 현금흐름 관련 핵심 사항
            - 예외적/특이사항

            4. 기업 및 산업 분석
            - 계정과목 구성 기반 기업 특성 파악
            - 자산/부채 구조 기반 산업 특성 분석
            - 매출/이익 패턴 기반 사업 모델 추정

            출력 형식:
            {
            \"분석결과\": {
                \"핵심요약\": {
                \"주요포인트\": [...],
                \"주요트렌드\": [...]
                },
                \"주요지표\": {
                \"매출이익\": {...},
                \"재무상태\": {...},
                \"주요비율\": {...}
                },
                \"세부인사이트\": {
                \"수익성\": \"...\",
                \"안정성\": \"...\",
                \"현금흐름\": \"...\",
                \"특이사항\": \"...\"
                },
                \"기업분석\": {
                \"추정기업명\": \"...\",
                \"신뢰도\": \"(상/중/하)\",
                \"추정근거\": \"...\"
                },
                \"산업분석\": {
                \"추정산업\": \"...\",
                \"신뢰도\": \"(상/중/하)\",
                \"추정근거\": \"...\"
                }
            }
            }

            주요 가이드라인:
            - 모든 금액은 천원/백만원 단위 사용 (천단위 구분자 포함)
            - 비율은 소수점 1자리까지 표시 (예: 23.4%)
            - 증감은 상대값과 절대값 모두 표시 (예: +12.3%, +1.2억원)
            - 중요도에 따라 정보 구조화
            - 모든 판단은 객관적 데이터에 근거
            - 추정의 불확실성은 명확히 표시
            - 일회성/경상성 항목 구분하여 설명
            - 산업 특성 반영한 해석 제공"""
            },
            {
                "role": "user",
                "content": f"가능한 많은 주요 정보를 포함하여 다음 요약을 작성하시오: {context}"
            }
        ]

    def _prepare_request_data(self, context, max_tokens):
        """요청 데이터 준비"""
        return {
            'messages': self._prepare_messages(context),
            'topP': 0.6,
            'topK': 0,
            'maxTokens': max_tokens,
            'temperature': 0.1,
            'repeatPenalty': 2.0,
            'stopBefore': [],
            'includeAiFilters': False,
            'seed': 0
        }

    def _send_request(self, request_data):
        """스트리밍 API 요청 전송"""
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        response_text = ""
        try:
            with requests.post(
                f"{self.host}{self.endpoint}",
                headers=headers,
                json=request_data,
                stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        # JSON 응답 파싱 및 텍스트 누적
                        decoded_line = line.decode('utf-8')
                        response_text += self._parse_stream_response(decoded_line)
                        
            return response_text.strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            raise

    def _parse_stream_response(self, line):
        """스트리밍 응답 파싱"""
        try:
            data = json.loads(line)
            if 'error' in data:
                logging.error(f"Error in response: {data['error']}")
                raise Exception(data['error'])
            return data.get('text', '')
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse response: {str(e)}")
            return ''

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=100, stop_sequence=None):
        """
        텍스트 요약 수행
        Args:
            context (str): 요약할 텍스트
            max_tokens (int): 최대 토큰 수 (기본값: 100)
            stop_sequence (str, optional): 중단 시퀀스
        Returns:
            str: 요약된 텍스트
        """
        try:
            # 입력 길이 검증
            input_length = len(context)
            if input_length > 35000:
                error_msg = f"Input text too long: {input_length} chars (Max: 35000)"
                logging.error(error_msg)
                return error_msg

            # 요청 데이터 준비 및 전송
            request_data = self._prepare_request_data(context, max_tokens)
            summary = self._send_request(request_data)

            # 결과 로깅
            logging.info(
                f"HyperCLOVA X 003 Summarization - "
                f"Input length: {input_length} chars, "
                f"Output length: {len(summary)} chars"
            )

            return summary

        except Exception as e:
            logging.error(f"Summarization failed: {str(e)}")
            return str(e)