from typing import List
from dotenv import load_dotenv
from llama_cloud import RetrievalMode
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
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, KnowledgeGraphIndex, PropertyGraphIndex

from .retriever_kg import CustomNeo4jRetriever
from ..prompt import get_query_gen_prompt
from ...setting import RAGSettings

from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from llama_index.core.indices.knowledge_graph.retrievers import KGRetrieverMode

class TwoStageRetriever(QueryFusionRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        setting: RAGSettings | None = None,
        llm: str | None = None,
        query_gen_prompt: str | None = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = ...,
        num_queries: int = 4,
        use_async: bool = True,
        verbose: bool = False,
        callback_manager: CallbackManager | None = None,
        objects: List[IndexNode] | None = None,
        object_map: dict | None = None,
        retriever_weights: List[float] | None = None
    ) -> None:
        super().__init__(
            retrievers, llm, query_gen_prompt, mode, similarity_top_k, num_queries,
            use_async, verbose, callback_manager, objects, object_map, retriever_weights
        )
        self._setting = setting or RAGSettings()
        self._rerank_model = SentenceTransformerRerank(
            top_n=self._setting.retriever.top_k_rerank,
            model=self._setting.retriever.rerank_llm,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)
        results = self._simple_fusion(results)
        return self._rerank_model.postprocess_nodes(results, query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)
        results = self._simple_fusion(results)
        return self._rerank_model.postprocess_nodes(results, query_bundle)