from llama_index.core.base.llms.types import ChatMessage
from setting.setting import RAGSettings
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine
import asyncio
import re

class LLMKnowledgeGraphBuilder:    
    setting: RAGSettings
    
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self.setting = setting or RAGSettings()
        self.model_name = "deepseek-r1:7b"
        self._init_model()
        self.debug = True
    
    def _init_model(self, host="localhost"):
            setting = self.setting
            
            with open("./prompt.txt", "r") as f:
                system_prompt = f.read()
        
            settings_kwargs = {
                "tfs_z": setting.ollama.tfs_z,
                "top_k": setting.ollama.top_k,
                "top_p": setting.ollama.top_p,
                "repeat_last_n": setting.ollama.repeat_last_n,
                "repeat_penalty": setting.ollama.repeat_penalty,
            }
            
            Settings.llm = Ollama(
                model=self.model_name,
                system_prompt=system_prompt,
                base_url=f"http://{host}:{setting.ollama.port}",
                temperature=setting.ollama.temperature,
                context_window=setting.ollama.context_window,
                request_timeout=999_999, # Requests may take long depending on model
                additional_kwargs=settings_kwargs
            )  
    
    async def build_knowledge_graph_ttl(self, file_name: str, split_content: list[str])-> str:
        if len(split_content) == 0:
            return ""
        
        with open("./prompt_flint.txt", "r") as f:
            flint_prompt = f.read()
            
        engine = SimpleChatEngine.from_defaults(
            llm = Settings.llm,
            memory = ChatMemoryBuffer(
                    token_limit=100_000
                )
        )
            
        first_msg = await engine.astream_chat(message=flint_prompt)
        
        async for token in first_msg.async_response_gen():
            print(token, end="", flush=True)
            
        messages_to_send= [ChatMessage(role="user", content=first_msg.response)]        
        final_file_content = ""         
            
        for part in split_content:            
            result = await engine.astream_chat(message=part, chat_history=messages_to_send)
            async for token in result.async_response_gen():
                if (self.debug):
                    print(token, end="", flush=True)
                    
            filtered_content = self._extract_content(result.response)
            final_file_content += (filtered_content + "\n\n\n\n")
            
        return final_file_content
                
    def _extract_content(self, model_output: str):
        ttl_pattern = re.compile(r"(@prefix\s+.*?\n)(.*?)(?=\n\n|\Z)", re.DOTALL)
        
        # Remove the <think> part if it exists
        cleaned_output = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL)
        
        # Extract the TTL content
        match = ttl_pattern.search(cleaned_output)
        if match:
            # Return the TTL content, including the prefix declarations
            return match.group(1) + match.group(2).strip()
        else:
            # If no TTL content is found, return an empty string or raise an error
            return ""


if __name__ == "__main__":
    async def main():
        i = LLMKnowledgeGraphBuilder()
        await i.build_knowledge_graph_ttl()
    
    asyncio.run(main())