from typing import List, Tuple, Dict, Union
import re
import raptor.config.api_config as api_config


class FinQATokenizer:
    """
    """
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size  # byte 단위

    def encode(self, node_texts):
        """
        """
        return node_texts

    def decode(self, token_pairs: List[Tuple[Dict, str]]) -> List[Tuple[Dict, str]]:
        """
        """
        return token_pairs

    def __str__(self):
        return "FinQATokenizer"