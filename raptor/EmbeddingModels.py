import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import http.client
import json
import time

from tenacity import retry, stop_after_attempt, wait_random_exponential
from .config import (
    CLOVA_HOST, 
    CLOVA_API_KEY, 
    CLOVA_API_GATEWAY_KEY, 
    EMBEDDING_CONFIG
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

class HyperCLOVAEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.host = CLOVA_HOST
        self.api_key = CLOVA_API_KEY
        self.api_key_primary_val = CLOVA_API_GATEWAY_KEY
        self.request_id = EMBEDDING_CONFIG['request_id']
        self.endpoint = EMBEDDING_CONFIG['endpoint']
        self.last_request_time = 0
        self.min_request_interval = 1  # 최소 요청 간격 (초)

    def _check_rate_limits(self, response):
        """응답 헤더에서 rate limit 정보를 확인하고 처리"""
        headers = response.getheaders()
        rate_limits = {}
        
        for key, value in headers:
            if key.startswith('x-ratelimit-'):
                rate_limits[key] = value
        
        remaining_requests = int(rate_limits.get('x-ratelimit-remaining-requests', 0))
        remaining_tokens = int(rate_limits.get('x-ratelimit-remaining-tokens', 0))
        reset_requests = rate_limits.get('x-ratelimit-reset-requests', '0s').replace('s', '')
        reset_tokens = rate_limits.get('x-ratelimit-reset-tokens', '0s').replace('s', '')
        
        # 로깅을 통한 모니터링
        logging.info(f"Rate Limits - Remaining Requests: {remaining_requests}, "
                    f"Remaining Tokens: {remaining_tokens}, "
                    f"Reset Requests in: {reset_requests}s, "
                    f"Reset Tokens in: {reset_tokens}s")
        
        return remaining_requests, remaining_tokens, float(reset_requests), float(reset_tokens)

    def _wait_if_needed(self, remaining_requests, remaining_tokens, reset_requests, reset_tokens):
        """rate limit에 따른 대기 시간 계산 및 적용"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if remaining_requests <= 1 or remaining_tokens <= 1:
            wait_time = max(float(reset_requests), float(reset_tokens))
            logging.warning(f"Rate limit near threshold. Waiting for {wait_time} seconds.")
            time.sleep(wait_time + 0.1)  # 여유를 두고 대기
        elif elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            time.sleep(wait_time)
        
        self.last_request_time = time.time()

    def _send_request(self, text):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id
        }
        
        request_data = {
            "text": text.replace("\n", " ")
        }
        
        conn = http.client.HTTPSConnection(self.host)
        conn.request(
            'POST',
            self.endpoint,
            json.dumps(request_data),
            headers
        )
        
        response = conn.getresponse()
        
        # rate limit 체크 및 처리
        remaining_requests, remaining_tokens, reset_requests, reset_tokens = self._check_rate_limits(response)
        self._wait_if_needed(remaining_requests, remaining_tokens, reset_requests, reset_tokens)
        
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        try:
            response = self._send_request(text)
            
            if response['status']['code'] == '20000':
                embedding = response['result']['embedding']
                logging.info(f"HyperCLOVA Embedding - Input length: {len(text)}, "
                           f"Output dimension: {len(embedding)}")
                return embedding
            else:
                logging.error(f"Error in HyperCLOVA API: {response}")
                raise Exception(f"Error: {response['status']['message']}")
                
        except Exception as e:
            logging.error(f"Exception in HyperCLOVA embedding creation: {e}")
            if "429" in str(e):  # Rate limit exceeded
                logging.warning("Rate limit exceeded. Implementing exponential backoff...")
                raise  # retry decorator will handle this
            raise e