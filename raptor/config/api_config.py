# OPENAI_API
OPENAI_API_KEY = "여기에_OPENAI_API_키_입력"

# 하이퍼클로바
CLOVA_HOST = 'clovastudio.apigw.ntruss.com'  # 고정
CLOVA_API_KEY = '여기에_CLOVA_API_키_입력'
CLOVA_API_GATEWAY_KEY = '여기에_CLOVA_GATEWAY_키_입력'

# 요약 모델 설정
SUMMARY_CONFIG = {
    'request_id': '여기에_REQUEST_ID_입력',
    'endpoint': '/testapp/v1/api-tools/summarization/v2/e996884683d148c591bf295c7b4b8fb3'  # 고정
}

# 임베딩 모델 설정
EMBEDDING_CONFIG = {
    'request_id': '여기에_REQUEST_ID_입력',
    'endpoint': '/testapp/v1/api-tools/embedding/clir-emb-dolphin/7437fa9facdf4658bcfc4c4d1cd4ebb5'  # 고정
}

EXTRACT_CONFIG = {
    'request_id': '여기에_REQUEST_ID_입력',
    'endpoint': '/testapp/v1/chat-completions/HCX-003'  # 고정
}

HCX_003_CONFIG = {
    'request_id': '여기에_REQUEST_ID_입력',
    'endpoint': '/testapp/v1/chat-completions/HCX-003'  # 고정
}

HCX_SEGMENTATION_CONFIG = {
    'request_id': '여기에_REQUEST_ID_입력',
    'endpoint': '/testapp/v1/api-tools/segmentation/818efcf6224c4eb39c967ebd910b3841'  # 고정
}

# 토큰 제한 설정
HCX_EMBEDDING_MAX_TOKENS = 8191
OPENAI_EMBEDDING_MAX_TOKENS = 8191