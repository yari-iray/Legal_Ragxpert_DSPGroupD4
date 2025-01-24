
import re
from typing import Any, List
import uuid
from dotenv import load_dotenv
from llama_index.core.retrievers import (
    BaseRetriever,
)


from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.llms.llm import LLM

from llama_index.core import Settings, KnowledgeGraphIndex, PropertyGraphIndex
from sympy import Q

from ..prompt import get_query_gen_prompt
from ...setting import RAGSettings

from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext

from llama_index.core import PromptTemplate

load_dotenv()

class _Connectivity:
    _metadata: Any | None = None
    
    @staticmethod
    def get_metadata(neo4j: Neo4jGraphStore):
        if _Connectivity._metadata:
            return _Connectivity._metadata
        
        query = "CALL apoc.meta.data()"
        result = neo4j.query(query)
        _Connectivity._metadata = result
        
        return result

        
REFINEMENT_PROMPT = PromptTemplate("""
        You are a helpful assistant that refines search cypher queries for a graph database.
        The original query is: "{query}".
        Provide a refined cypher query that is more likely to retrieve relevant results.)"""
    )

class CustomNeo4jRetriever(BaseRetriever):
    def __init__(self, graph_store: Neo4jGraphStore, llm: LLM | None = None, num_queries = 5):
        assert num_queries > 0
        
        self.graph_store = graph_store
        self.metadata = _Connectivity.get_metadata(self.graph_store)
        self.llm = llm or Settings.llm        
        
        assert self.llm is not None
        self.num_queries = num_queries

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes from the Neo4j graph store based on the query.
        """
        user_query = query_bundle.query_str
        
        # Execute the cypher query
        cypher_query = self._generate_cypher_query(user_query)
        results = self.graph_store.query(cypher_query)
        
        # todo, generate list of nodes
        nodes: list[NodeWithScore] = []
        for record in results:
            n = TextNode(
                id_ = str(uuid.uuid4()),
                text = str(record)
            ) # type: ignore
            
            nodes.append(
                NodeWithScore(
                    node=n, 
                    score= self._simple_calculate_relevance(record, user_query)
                )
            )    
            
        ranked_nodes = sorted(
            nodes,
            key=lambda node: node.score or 0,
            reverse=True
        )          

        # Step 4: Rerank the results using the LLM    
        # TODO: Implement scores
        ranked_nodes = self._rank_results_with_llm(ranked_nodes, user_query)
        return ranked_nodes
    
    def _simple_calculate_relevance(self, node: dict, user_query: str) -> float:
        """
        Extremely basic ranking
        """
        relevance = 0.0
        for key, value in node.items():
            if isinstance(value, str) and any(keyword.lower() in value.lower() for keyword in user_query.split()):
                relevance += 1.0

        return relevance
    
    def _generate_cypher_query(self, query: str) -> str:
        """
        Use the LLM to convert a natural language query into a Cypher query.
        """
        prompt = f"""
        You are a helpful assistant that converts natural language queries into Cypher queries for a Neo4j graph database.
        The graph contains nodes with properties like `name`, `description`, and `type`.
        
        The neo4j graph contains the following node labels and properties:
        {self.metadata}
        
        Convert the following natural language query into a Cypher query:
        "{query}"
        Return only the Cypher query.
        """
        cypher_query = self.llm.complete(prompt).text.strip()
        return cypher_query


    # def _refine_query_with_llm(self, query: str) -> str:
    #     """
    #     Use the LLM to refine the query for better retrieval.
    #     """
    #     p = REFINEMENT_PROMPT.format(query=query)        

    #     refined_query = self.llm.complete(p).text
    #     return refined_query.strip()
    
    # def _deduplicate_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
    #     """
    #     Deduplicate nodes based on their IDs.
    #     """
    #     seen_ids = set()
    #     unique_nodes = []
    #     for node in nodes:
    #         if node.node.id not in seen_ids:
    #             seen_ids.add(node.node.id)
    #             unique_nodes.append(node)
    #     return unique_nodes
    
    RERANK_PROMPT = PromptTemplate("""
        You are a helpful assistant that reranks search results based on relevance to the query.
        Your task is to rank the nodes based on relevance and return the node ids in the order of relevance.
        Never output anything else besides the node ids.
        The query is: "{query}".
        Here are the retrieved results:
        {nodes}
        
        You are going to rerank the nodes based on relevance to the query.
        Output only the reranked node ids separated by a comma, do not include your reasoning or thinking process.
        """)
    
    uuid4_pattern = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}")

    def _rank_results_with_llm(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """
        Use the LLM to rerank the retrieved nodes based on relevance to the query.
        """        
        
        interpretable_nodes = [ f"node_id = {n.node_id}, content = {n.text}" for n in nodes ]        
        reranking_prompt = self.RERANK_PROMPT.format(query=query, nodes=interpretable_nodes)
        
        split_llm_output = self.llm.complete(reranking_prompt).text\
            .strip()\
            .split(",")

        # Reorder the nodes based on the LLM's reranking
        # This is to ensure that all nodes are output in case the llm outputs a bad response
        nodes_copy = nodes.copy()
        
        reranked_nodes = []
        
        for node_id in split_llm_output:
            id_regex_match = self.uuid4_pattern.search(node_id)
            
            if not id_regex_match:
                continue
            
            match = id_regex_match.group(0)
            for node in nodes:
                if str(node.node.node_id) != match:
                    continue
                
                if not node in nodes:
                    reranked_nodes.append(node)
                    nodes_copy.remove(node)
                break
                    
        # Some nodes have not been reranked by the LLM, put them as the last nodes
        if len(nodes_copy) > 0:
            reranked_nodes.extend(nodes_copy)

        return reranked_nodes