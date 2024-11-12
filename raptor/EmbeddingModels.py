import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import http.client
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential
from config import (
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
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        try:
            response = self._send_request(text)
            
            if response['status']['code'] == '20000':
                return response['result']['embedding']
            else:
                logging.error(f"Error in HyperCLOVA API: {response}")
                raise Exception(f"Error: {response['status']['message']}")
                
        except Exception as e:
            logging.error(f"Exception in HyperCLOVA embedding creation: {e}")
            raise e