import PyPDF2
from tqdm import tqdm
import matplotlib.pyplot as plt
from spacy.tokens import Doc
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
import logging
import re
import fitz
from llama_index.core.schema import BaseNode
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from ...setting import RAGSettings
from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from rdflib import Graph
from setting.setting import RAGSettings
from core.ingestion.kg_builder import LLMKnowledgeGraphBuilder

class DocumentAnalyzer:
    def __init__(self, setting: RAGSettings | None = None):
        self._setting = setting or RAGSettings()        
        self.logger = logging.Logger("doc-analyzer")
        self._stored_filenames = set()
        self.kg_builder = LLMKnowledgeGraphBuilder(self._setting)
        
    
    def add_to_neo4j_db(self, path_or_content: str):
        settings = self._setting

        auth_data = {
            'uri': settings.neo4j.url,
            'database': settings.neo4j.database,
            'user': settings.neo4j.username,
            'pwd': settings.neo4j.password
        }

        config = Neo4jStoreConfig(auth_data=auth_data,                        
            handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE, batching=True)

        neo4j_aura = Graph(store=Neo4jStore(config=config))
        
        # Calling the parse method will implictly open the store
        neo4j_aura.parse(path_or_content, format="ttl")
        neo4j_aura.close(True)    

    
    def _filter_text(self, text):
        # Define the regex pattern.
        pattern = r'[a-zA-Z0-9 \u00C0-\u01B0\u1EA0-\u1EF9`~!@#$%^&*()_\-+=\[\]{}|\\;:\'",.<>/?]+'
        matches = re.findall(pattern, text)
        # Join all matched substrings into a single string
        filtered_text = ' '.join(matches)
        # Normalize the text by removing extra whitespaces
        normalized_text = re.sub(r'\s+', ' ', filtered_text.strip())

        return normalized_text

    def store_nodes(self, input_files: list[str]) -> None:
        if len(input_files) == 0:
            return
        
        text_by_file_name: dict[str, list[str]] = {}
            
        for input_file in tqdm(input_files, desc="Ingesting data"):
            file_name = input_file.strip().split('/')[-1]
            
            if file_name in self._stored_filenames:
                continue
            
            document = fitz.open(input_file)
            text_per_page = []
            for _, page in enumerate(document):
                page_text = page.get_text("text")
                page_text = self._filter_text(page_text)
                text_per_page.append(page_text)
                
            text_by_file_name[file_name] = text_per_page
            self._stored_filenames.add(file_name)
            
        
        for file_name, file_content in text_by_file_name.items():
            result_file_content = self.kg_builder.build(file_name, file_content)
            self.add_to_neo4j_db(result_file_content)
            
        
