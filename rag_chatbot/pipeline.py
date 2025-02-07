from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    KgDataIngestion,
    LocalKgModels,
    LocalEmbedding,
    get_system_prompt,
)
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse, BaseChatEngine
from llama_index.core.prompts import ChatMessage, MessageRole



class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._language = "eng"
        self._model_name = ""
        self._system_prompt = get_system_prompt(is_rag_prompt=False)
        self._engine = LocalChatEngine(host=host)
        self._default_model = LocalKgModels.set(self._model_name, host=host)
        self._query_engine: BaseChatEngine = None
        self._ingestion = KgDataIngestion()
        Settings.llm = LocalKgModels.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)

    def get_model_name(self):
        return self._model_name

    def set_model_name(self, model_name: str):
        self._model_name = model_name

    def get_language(self):
        return self._language

    def set_language(self, language: str):
        self._language = language

    def get_system_prompt(self):
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str | None = None):
        self._system_prompt = system_prompt or get_system_prompt(
            is_rag_prompt=self._ingestion.check_nodes_exist()
        )

    def set_model(self):
        Settings.llm = LocalKgModels.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            host=self._host,
        )
        self._default_model = Settings.llm

    def reset_engine(self, mode):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model, mode=mode
        )

    def reset_documents(self):
        self._ingestion.reset()

    def clear_conversation(self):
        self._query_engine.reset()

    def reset_conversation(self, mode = "kg"):
        self.reset_engine(mode)
        self.set_system_prompt(
            get_system_prompt(is_rag_prompt=False)
        )

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name, self._host)

    def pull_model(self, model_name: str):
        return LocalKgModels.pull(self._host, model_name)

    def pull_embed_model(self, model_name: str):
        return LocalEmbedding.pull(self._host, model_name)

    def check_exist(self, model_name: str) -> bool:
        return LocalKgModels.check_model_exists(self._host, model_name)

    def check_exist_embed(self, model_name: str) -> bool:
        return LocalEmbedding.check_model_exist(self._host, model_name)

    def store_nodes(self, input_files: list[str] | None = None) -> None:
        input_files = input_files or []        
        self._ingestion.store_nodes(input_files=input_files)

    def set_chat_mode(self, system_prompt: str | None = None):
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()

    def set_engine(self, mode = "kg"):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            mode=mode,
        )

    def get_history(self, chatbot: list[list[str]]):
        history = []
        for chat in chatbot:
            if chat[0]:
                history.append(ChatMessage(role=MessageRole.USER, content=chat[0]))
                history.append(ChatMessage(role=MessageRole.ASSISTANT, content=chat[1]))
        return history

    def query(
        self, mode: str, message: str, chatbot: list[list[str]]
    ) -> StreamingAgentChatResponse:       
        if mode == "chat":
            history = self.get_history(chatbot)
            return self._query_engine.stream_chat(message, history)
        elif mode == "kg":
            self._query_engine.reset()
            return self._query_engine.stream_chat(message)
        
        raise NotImplementedError()
