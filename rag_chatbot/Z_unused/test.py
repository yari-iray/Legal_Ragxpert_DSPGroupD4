import os
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

def test_embedding_is_not_broken():
    test = HuggingFaceEmbedding(
                    model_name="BAAI/bge-large-en-v1.5",

                    cache_folder=os.path.join(os.getcwd(), "data/huggingface"),
                    trust_remote_code=True,
                )
    
if __name__ == "__main__":
    test_embedding_is_not_broken()