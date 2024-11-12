import logging
import os

from openai import OpenAI
import json
import http

import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer
from .config import (
    CLOVA_HOST, 
    CLOVA_API_KEY, 
    CLOVA_API_GATEWAY_KEY, 
    HCX_003_CONFIG,
)

class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]

class HCX_003_QAModel(BaseQAModel):
    def __init__(self):
        """
        HyperCLOVA X 모델을 사용하는 QA 모델 초기화
        """
        self.api_key = CLOVA_API_KEY
        self.api_gateway_key = CLOVA_API_GATEWAY_KEY
        self.host = CLOVA_HOST
        self.request_id = HCX_003_CONFIG['request_id']
        self.endpoint = HCX_003_CONFIG['endpoint']

    def generate_search_question(self, question):
        """
        주어진 질문을 기반으로 검색에 최적화된 질문을 생성
        
        Args:
            question (str): 원본 질문
            
        Returns:
            str: 검색용으로 변환된 질문
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """당신은 검색 최적화 전문가입니다. 주어진 질문을 분석하여, 해당 질문에 답변하기 위해 필요한 정보를 찾는데 최적화된 검색 문장을 생성해주세요. 
                    핵심 키워드와 중요 정보를 포함하되, 검색에 효과적인 형태로 변환해주세요.

                    예시:
                    1. 원본 질문: "회사의 부채비율은 어떻게 되나요?"
                    검색 문장: "부채비율은 RATIO% 수준"

                    2. 원본 질문: "영업이익률이 어떻게 변화했나요?"
                    검색 문장: "영업이익률은 전년 대비"

                    3. 원본 질문: "주요 매출 성장 동력은 무엇인가요?"
                    검색 문장: "매출 성장은 주로 사업부문이"

                    4. 원본 질문: "연구개발 투자는 얼마나 하고 있나요?"
                    검색 문장: "연구개발비는 XXX억원을 투자"

                    5. 원본 질문: "현금흐름이 어떻게 되나요?"
                    검색 문장: "영업활동 현금흐름은"

                    6. 원본 질문: "배당 정책은 어떻게 되나요?"
                    검색 문장: "배당성향은 수준을 유지"

                    7. 원본 질문: "신규 사업 계획이 있나요?"
                    검색 문장: "신규 사업으로 추진"

                    8. 원본 질문: "ESG 활동은 어떻게 하고 있나요?"
                    검색 문장: "ESG 경영을 위해"

                    주의사항:
                    - 재무제표나 사업보고서에서 흔히 사용되는 문구 스타일로 변환
                    - 불필요한 수식어 제거
                    - 핵심 키워드를 문장 앞쪽에 배치
                    - 재무 용어는 공시 자료에서 사용하는 정확한 용어로 변환
                    """
                },
                {
                    "role": "user",
                    "content": f"다음 질문에 답변하기 위해 필요한 정보를 찾기 위한 검색 문장을 생성해주세요. 답변은 재무제표에 있길 원하는 원본 문장과 유사하도록 출력하세요.\n\n질문: {question}"
                }
            ]
            
            response = self._send_request(messages)
            
            if response['status']['code'] == '20000':
                search_question = response['result']['message']['content'].strip()
                return search_question
            else:
                logging.error(f"Error in search question generation: {response}")
                return question
                
        except Exception as e:
            logging.error(f"Exception in search question generation: {e}")
            return question

    def _send_request(self, messages):
        """
        CLOVA Studio API에 요청을 보내는 내부 메서드
        """
        headers = {
            'Content-Type': 'application/json',
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_gateway_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id
        }
        
        request_data = {
            "messages": messages,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 256,
            "temperature": 0.3,
            "repeatPenalty": 5.0,
            "stopBefore": [],
            "includeAiFilters":False
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

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question):
        """
        주어진 컨텍스트를 기반으로 질문에 답변
        
        Args:
            context (str): 참고할 문맥 정보
            question (str): 답변할 질문
            
        Returns:
            str: 생성된 답변
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "당신은 전문 지식을 가진 금융 분석가입니다. 주어진 문맥을 기반으로 질문에 대해 정확하고 간단명료하게 답변해주세요. 불필요한 설명은 제외하고, 핵심적인 정보만 제공해주세요. 여러 숫자 데이터들을 혼동하지 말고, 연도와 액수 이러한 데이터들을 명확히 인지하고 답변하세요."
                },
                {
                    "role": "user",
                    "content": f"다음 정보를 참고하여 질문에 답변해주세요:\n\n문맥: {context}\n\n질문: {question}"
                }
            ]
            
            response = self._send_request(messages)
            
            if response['status']['code'] == '20000':
                return response['result']['message']['content'].strip()
            else:
                logging.error(f"Error in CLOVA Studio API: {response}")
                raise Exception(f"Error: {response['status']['message']}")
                
        except Exception as e:
            logging.error(f"Exception in HCX-003 question answering: {e}")
            raise e