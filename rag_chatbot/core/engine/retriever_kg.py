
import re
from typing import Any, List
import uuid
from dotenv import load_dotenv
from llama_index.core.retrievers import (
    BaseRetriever,
)
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.llms.llm import LLM
from llama_index.core import Settings
from ...setting import RAGSettings
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.postprocessor import SentenceTransformerRerank

load_dotenv()

class _Connectivity:
    _metadata: Any | None = None
    
    @staticmethod
    def get_metadata(graph_store: Neo4jGraphStore):        
        if _Connectivity._metadata:
            return _Connectivity._metadata
        
        query = "CALL apoc.meta.data()"
        result = graph_store.query(query)
        _Connectivity._metadata = result
        
        return result

class CustomNeo4jRetriever(BaseRetriever):
    def __init__(self, graph_store: Neo4jGraphStore, llm: LLM | None = None, setting: RAGSettings | None = None):
        self._setting = setting or RAGSettings()
        self._rerank_model = SentenceTransformerRerank(
            top_n=self._setting.retriever.top_k_rerank,
            model=self._setting.retriever.rerank_llm,
        )
        
        self._graph_store = graph_store
        self._metadata = _Connectivity.get_metadata(self._graph_store)
        self._llm = llm or Settings.llm
        
        assert self._llm is not None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes from the Neo4j graph store based on the query.
        """
        user_query = query_bundle.query_str
        
        # Step 1: Build cypher query
        cypher_query = self._generate_cypher_query(user_query)
        
        # Step 2: Execute cypher query
        results: list[dict] = self._graph_store.query(cypher_query)
        
        # Step 3: Do basic ranking
        
        keywords = [x.lower() for x in user_query.split()]
        nodes: list[NodeWithScore] = []
        for record in results:
            text_node = TextNode(
                id_ = str(uuid.uuid4()),
                text = str(record)
            )
            
            nodes.append(
                NodeWithScore(
                    node=text_node, 
                    score= self._simple_calculate_relevance(record, keywords)
                )
            )

        # Step 4: Rerank the results using our reranking model
        ranked_nodes = self._rerank_nodes(query_bundle, nodes)

        return ranked_nodes
    
    def _simple_calculate_relevance(self, node: dict, keywords: list[str]) -> float:
        """
        Naive ranking according to if query text is contained in a result
        """
        relevance = 0.0        
        
        for value in node.values():
            if isinstance(value, str) and any(keyword in value.lower() for keyword in keywords):
                relevance += 1.0

        return relevance
    
    def _generate_cypher_query(self, query: str) -> str:
        """
        Use the LLM to convert a natural language query into a Cypher query.
        """
        cypher_generation_prompt = f"""
        You are a helpful assistant that converts natural language queries into Cypher queries for a Neo4j graph database.
        The graph contains nodes with properties like `name`, `description`, and `type`.
        
        The neo4j graph contains the following node labels and properties:
        {self._metadata}
        
        Convert the following natural language query into a Cypher query:
        "{query}"
        Return only the Cypher query.
        """
        cypher_query = self._llm.complete(cypher_generation_prompt).text.strip()
        return cypher_query
    
    def _rerank_nodes(self, query_bundle: QueryBundle, nodes: list[NodeWithScore]):
        """
        Rerank the retrieved nodes using our reranking model
        """
        ranked = self._rerank_model.postprocess_nodes(nodes, query_bundle)
        return ranked