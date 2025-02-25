from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode
from typing import List
from .provider import LocalRetrieverProvider
from ...setting import RAGSettings


class LocalChatEngine:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        self._setting = setting or RAGSettings()
        self._retrieverprovider = LocalRetrieverProvider(self._setting)
        self._host = host

    def set_engine(
        self,
        llm: LLM,
        mode: str = "kg",
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:

        # Normal chat engine
        if mode == "chat":
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.ollama.chat_token_limit
                )
            )

        elif mode == "kg":
            # Chat engine with documents
            mem = ChatMemoryBuffer(
                    token_limit=self._setting.ollama.chat_token_limit
                )
            
            retriever = self._retrieverprovider.get_retriever(llm=llm)
            return CondensePlusContextChatEngine.from_defaults(
                retriever=retriever,
                llm=llm,
                memory=mem
            )
            
        raise NotImplementedError()
