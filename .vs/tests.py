import glob
from typing import Any, List, Type, TypeVar
import uuid
from dotenv import load_dotenv
from llama_cloud import RetrievalMode, TextNode
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    # VectorIndexRetriever,    
    KnowledgeGraphRAGRetriever,
    RouterRetriever,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, IndexNode
from llama_index.core.llms.llm import LLM
from llama_index.core import Settings, KnowledgeGraphIndex, PropertyGraphIndex
from sympy import Q


from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from llama_index.core.indices.knowledge_graph.retrievers import KGRetrieverMode
from llama_index.core import PromptTemplate
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode

def _simple_calculate_relevance(record, user_query):
    return 1

user_query = "test de user query"
results = [{"test": "het werkt"}]

nodes: list[NodeWithScore] = []
for record in results:
    n = TextNode(
        id_ = str(uuid.uuid4()),
        text = str(record)
    ) # type: ignore
    
    nodes.append(
        NodeWithScore(
            node=n, 
            score= _simple_calculate_relevance(record, user_query)
        )
    )
    
