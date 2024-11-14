import logging
import os
from abc import ABC, abstractmethod
import http.client
import json
import requests

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
        prompts = [
            {
                "role": "system",
                "content": """
                    당신은 재무 데이터 분석 전문가입니다.
                    주어진 재무 데이터를 아래 지침에 따라 종합적으로 요약해주세요.

                    지시사항:
                    1. 모든 테이블과 텍스트 데이터를 꼼꼼히 분석
                    2. 데이터간 연관성을 찾아 맥락을 구성
                    3. 중요한 수치와 키워드는 반드시 포함
                    4. 시간 순서나 중요도 순서로 구성
                    5. 객관적 사실을 기반으로 서술
                    6. 필요한 경우 데이터의 의미나 영향도 설명

                    출력 형식:
                    [Overview]
                    - 전반적인 재무 상황 개요
                    - 주목할 만한 주요 변화나 특징
                    - 핵심 지표들의 전반적인 흐름

                    [Detailed Analysis]
                    - 테이블 데이터 분석
                    - 주요 수치와 그 의미
                    - 시계열적 변화와 패턴
                    - 항목간 관계성

                    - 텍스트 데이터 분석
                    - 주요 설명과 논점
                    - 중요 키워드와 그 맥락
                    - 특이사항이나 예외적 내용

                    [Key Insights]
                    - 발견된 주요 인사이트
                    - 데이터가 시사하는 의미
                    - 주목해야 할 특이사항
                    """
            },
            {
                "role": "user",
                "content": f"다음 재무 데이터를 종합적으로 분석하여 요약해주세요. 테이블과 텍스트의 모든 중요 정보를 포함해주시고, 데이터간 연관성을 찾아 의미 있는 맥락을 구성해주세요: {context}"
            }
        ]
        return prompts

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

    def _parse_sse_line(self, line: str) -> str:
        """SSE 라인 파싱 함수"""
        if not line:
            return ""
            
        try:
            # 'data: ' 접두어 확인 및 제거
            if line.startswith('data: '):
                line = line[6:]
            
            # [DONE] 메시지 처리    
            if line.strip() == '[DONE]':
                return ""
                
            # JSON 파싱
            data = json.loads(line)
            
            # 에러 체크
            if 'error' in data:
                self.logger.error(f"Error in response: {data['error']}")
                raise Exception(data['error'])
                
            # 메시지 추출
            if 'message' in data:
                return data['message'].get('content', '')
            
            return data.get('text', '')
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 오류: {str(e)}, 라인: {line}")
            return ""
        except Exception as e:
            self.logger.error(f"파싱 중 오류 발생: {str(e)}, 라인: {line}")
            return ""

    def _send_request(self, request_data):
        """스트리밍 API 요청 전송"""
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        full_response = []
        try:
            with requests.post(
                f"https://{self.host}{self.endpoint}",
                headers=headers,
                json=request_data,
                stream=True
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        content = self._parse_sse_line(decoded_line)
                        
                        if content:
                            print(content, end='', flush=True)  # 실시간 출력
                            full_response.append(content)
                            
            return ''.join(full_response).strip()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise

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
            input_length = len(str(context))
            MAX_CHAR_LENGTH = 20000
            
            if input_length > MAX_CHAR_LENGTH:
                self.logger.info(f"Input text too long: {input_length} chars (Max: {MAX_CHAR_LENGTH})")
                max_len = min(int(input_length * 0.8), int(MAX_CHAR_LENGTH * 0.8))
                context = str(context)[:max_len]
                self.logger.info(f"{max_len}자로 줄여서 요약을 진행합니다")

            # 요청 데이터 준비 및 전송
            request_data = self._prepare_request_data(context, max_tokens)
            summary = self._send_request(request_data)

            # 빈 응답 체크
            if not summary:
                raise ValueError("요약 결과가 비어있습니다")

            # 결과 로깅
            self.logger.info(
                f"HyperCLOVA X 003 Summarization - "
                f"Input length: {input_length} chars, "
                f"Output length: {len(summary)} chars"
            )

            return summary

        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            raise