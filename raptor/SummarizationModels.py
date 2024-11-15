import logging
import os
from abc import ABC, abstractmethod
import http.client
import json
import requests
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from .config.api_config import (
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
    """HyperCLOVA X 003 모델을 사용한 요약 클래스"""
    
    def __init__(self):
        self.host = CLOVA_HOST
        self.api_key = CLOVA_API_KEY
        self.api_key_primary_val = CLOVA_API_GATEWAY_KEY
        self.request_id = HCX_003_CONFIG['request_id']
        self.endpoint = HCX_003_CONFIG['endpoint']
        self.logger = logging.getLogger(__name__)

    def _send_request(self, request_data):
        """CLOVA Studio API에 요청을 보내는 내부 메서드"""
        headers = {
            'Content-Type': 'application/json',
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id
        }
        
        conn = http.client.HTTPSConnection(self.host)
        conn.request(
            'POST',
            self.endpoint,
            json.dumps(request_data),
            headers
        )
        
        response = conn.getresponse()
        result = json.loads(response.read().decode('utf-8'))
        conn.close()
        return result

    def summarize(self, context: str, max_tokens: int = 512) -> str:
        """텍스트 요약 수행"""
        if not context:
            raise ValueError("Empty input context")

        # 입력 길이 검증 및 조정
        if len(str(context)) > 1000:
            context = str(context)[:1000]
            self.logger.info(f"요약 모델 입력을 1000자로 잘라서 진행합니다")

        # 요청 데이터 준비
        request_data = {
                'messages': [
                    {
                        "role": "system",
                        "content": """
                            당신은 재무 데이터 분석 전문가입니다.
                            주어진 재무 데이터를 아래 지침에 따라 종합적으로 요약해주세요.

                            ####
                            지시사항:
                            1. 모든 테이블과 텍스트 데이터를 꼼꼼히 분석
                            2. 데이터간 연관성을 찾아 맥락을 구성
                            3. 중요한 수치와 키워드는 반드시 포함
                            4. 시간 순서나 중요도 순서로 구성
                            5. 객관적 사실을 기반으로 서술
                            6. 필요한 경우 데이터의 의미나 영향도 설명

                            ####
                            출력 형식:
                            - 전반적인 재무 상황 개요
                            - 주목할 만한 주요 변화나 특징
                            - 핵심 지표들의 전반적인 흐름

                            - 테이블 데이터 분석
                            - 주요 수치와 그 의미
                            - 시계열적 변화와 패턴
                            - 항목간 관계성
                            - 텍스트 데이터 분석
                            - 주요 설명과 논점
                            - 중요 키워드와 그 맥락
                            - 특이사항이나 예외적 내용
                            - 발견된 주요 인사이트
                            - 데이터가 시사하는 의미
                            - 주목해야 할 특이사항
                            """
                    },
                    {
                        "role": "user",
                        "content": f"다음 재무 데이터를 종합적으로 분석하여 요약해주세요. 테이블과 텍스트의 모든 중요 정보를 포함해주시고, 데이터간 연관성을 찾아 의미 있는 맥락을 구성해주세요: {context}",
                    }
                ],
                'topP': 0.6,
                'topK': 0,
                'maxTokens': max_tokens,
                'temperature': 0.1,
                'repeatPenalty': 2.0,
                'stopBefore': [],
                'includeAiFilters': False,
                'seed': 0
            }

        try:
            response = self._send_request(request_data)
            
            if response['status']['code'] == '20000':
                return response['result']['message']['content'].strip()
            else:
                self.logger.error(f"Error in CLOVA Studio API: {response}")
                raise Exception(f"Error: {response['status']['message']}")
                
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            raise