from typing import List
from dotenv import load_dotenv
from llama_cloud import RetrievalMode
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    
    KnowledgeGraphRAGRetriever,
    RouterRetriever,
)

from llama_index.core.schema import BaseNode
from llama_index.core.llms.llm import LLM

from .retriever_kg import CustomNeo4jRetriever
from ...setting import RAGSettings

from llama_index.graph_stores.neo4j import Neo4jGraphStore



load_dotenv()


class LocalRetrieverProvider:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        self._setting = setting or RAGSettings()
        self._host = host
        
        db_setting = self._setting.neo4j
        self._graph_store = Neo4jGraphStore(db_setting.username, 
                                      db_setting.password, 
                                      db_setting.url,
                                      db_setting.database)

    def get_retriever(
        self,
        nodes: List[BaseNode] | None = None,
        llm: LLM | None = None,
    ):
        # Unused, might be useful in future implementations if moving to a hybrid implementations
        # with a kg and doc store
        
        return CustomNeo4jRetriever(self._graph_store, llm)
