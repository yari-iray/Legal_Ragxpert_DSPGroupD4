from llama_index.core.base.llms.types import ChatMessage
from ...setting import RAGSettings
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine
import re

class LLMKnowledgeGraphBuilder:    
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self.setting = setting or RAGSettings()
        self.model_name = "deepseek-r1:7b"
        self.base_path = "./rag_chatbot/prompts/"
        self.verbose = True     
        self._init_model()
        self.ttl_pattern = re.compile(r"```turtle([\s\S]*?)```")
        
    
    def _init_model(self, host="localhost"):
            setting = self.setting
            
            with open(f"{self.base_path}prompt.txt", "r") as f:
                system_prompt = f.read()
        
            settings_kwargs = {
                "tfs_z": setting.ollama.tfs_z,
                "top_k": setting.ollama.top_k,
                "top_p": setting.ollama.top_p,
                "repeat_last_n": setting.ollama.repeat_last_n,
                "repeat_penalty": setting.ollama.repeat_penalty,
            }
            
            self.ingest_model = Ollama(
                model=self.model_name,
                system_prompt=system_prompt,
                base_url=f"http://{host}:{setting.ollama.port}",
                temperature=setting.ollama.temperature,
                context_window=setting.ollama.context_window,
                request_timeout=999_999, # Requests may take long depending on model
                additional_kwargs=settings_kwargs
            )  
    
    def build(self, file_name: str, split_content: list[str])-> str:
        if len(split_content) == 0:
            return ""
        
        with open(f"{self.base_path}prompt_flint.txt", "r") as f:
            flint_prompt = f.read()
            
        engine = SimpleChatEngine.from_defaults(
            llm = self.ingest_model,
            memory = ChatMemoryBuffer(token_limit=100_000)
        )
            
        prompt_msg = engine.stream_chat(message=flint_prompt)        
        for token in prompt_msg.response_gen:
            if self.verbose:
                print(token, end="", flush=True)
            
        messages_to_send= [ChatMessage(role="user", content=prompt_msg.response)]        
        final_file_content = ""
            
        for part in split_content:            
            result = engine.stream_chat(message=part, chat_history=messages_to_send)
            for token in result.response_gen:
                if self.verbose:
                    print(token, end="", flush=True)
                    
            filtered_content = self._extract_content(result.response)
            final_file_content += (filtered_content + "\n\n\n\n")
            
        return final_file_content
                
    def _extract_content(self, model_output: str):
        # Remove deepseek's <think> part
        cleaned_output = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL)
        
        # extract ttl
        match = self.ttl_pattern.search(cleaned_output)
        
        if match:
            return match.group(1)
        else:
            return ""
