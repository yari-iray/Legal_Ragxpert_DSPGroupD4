from llama_index.core.base.llms.types import ChatMessage
from setting.setting import RAGSettings
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine
import asyncio

class LLMIngestion:    
    setting: RAGSettings
    
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self.setting = setting or RAGSettings()
        self.model_name = "deepseek-r1:14b"
        self._init_model()
    
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
                model="deepseek-r1:7b",
                system_prompt=system_prompt,
                base_url=f"http://{host}:{setting.ollama.port}",
                temperature=setting.ollama.temperature,
                context_window=setting.ollama.context_window,
                request_timeout=999_999,
                additional_kwargs=settings_kwargs
            )  
            
    def split_file(self, input_file, lines_per_chunk=50):
        chunks = []

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            
            for line in f:
                lines.append(line)
                if len(lines) == lines_per_chunk:
                    chunks.append(''.join(lines))
                    lines = []
            
            if lines:
                chunks.append(''.join(lines))       

        return chunks
    
    async def query(self):
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
            
        msgs= [token.message]
            
        split_lines = self.split_file("./example_case.txt")
        


        with open("./output.txt", "w") as f:           
            
            for part in split_lines:
                # content = flint_prompt + part
                
                # msg = ChatMessage(role="user", content = content)
                # result = await Settings.llm.astream_chat(messages=[msg])
                
                result = await engine.astream_chat(message=part, chat_history=msgs)
                async for token in result.async_response_gen():
                    print(token, end="", flush=True)
                f.write(result.response)
                f.write("\n")
                f.write("\n")
                f.write("\n")


if __name__ == "__main__":
    async def main():
        i = LLMIngestion()
        await i.query()
    
    asyncio.run(main())
    
    

"""DURING HIS HIGHLY successful pleadings in the Barcelona Traction case,1
Roberto Ago used a metaphor that would not be acceptable today. When he was referring to the Delagoa Bay case, he said that Delagoa was an old lady whose veil it was not advisable to lift.2 One could say that now Barcelona Traction has also become an old lady. The long time that has passed since the Court’s judgments, and my limited role in the pleadings, assisting Roberto Ago and Antonio Malintoppi, may allow me to revisit the case with the eyes of a scholar, in the context of the present study of landmark decisions on international law. Barcelona Traction was a holding company which controlled a number of companies producing electricity in Catalonia. It was declared bankrupt in 1948 by the Court of Reus, in Spain, because it had failed to pay interest on some bonds issued in sterling. The sum was a small amount of money and Barcelona Traction had not paid interest because the Spanish monetary authorities had refused to allow the company to export profits that it had made in Spain. As can happen in civil law countries, the bankruptcy judgment was given without hearing the company concerned,
which could have subsequently filed an opposition with the same Court. However,
Barcelona Traction did not make use of this remedy. The bankruptcy proceedings
continued and Barcelona Traction lost control of its subsidiary companies. As a
result of the bankruptcy, Barcelona Traction’s assets were transferred to a Spanish
company, Fuerzas Eléctricas de Cataluña SA (FECSA). There was a suspicion that the Spanish authorities had contrived Barcelona Traction’s bankruptcy. The strong man in FECSA was Juan March. He was closely linked to General Francisco Franco, whom he had aided in the Spanish Civil War, especially by providing him with air transport from the Canary Islands at the begin-
ning of the war.3 On the other hand, there was no evidence of corruption concerning the judge who had declared the bankruptcy. Also, the accounting firm Peat and
Marwick found, on consultation by Spain, that the evaluation of Barcelona Traction’s
assets in the bankruptcy proceedings had been fair.
Belgium contended that various Spanish authorities had acted in breach of Spain’s
obligations under international law and requested reparation. After years of diplomatic exchanges, which involved also Canada, the United Kingdom and the United
States, Belgium made an application to the ICJ. The case concerned a huge sum of
money and involved an unprecedented number of lawyers, some of them with the
role of monitoring the work of other lawyers. The representatives of the private
interests played an important part. The principal actors were FECSA on the Spanish
side and SOFINA on the Belgian side. Belgium first applied to the ICJ on 23 September 1958. The claim concerned reparation for the injury caused to the company Barcelona Traction and was espoused by Belgium because of the alleged Belgian nationality of the shareholders of the
company. Belgium filed a Memorial and Spain Preliminary Objections. The proceedings were discontinued by Belgium on 23 March 1961 pending negotiations for a settlement between representatives of the private interests. Spain declared that it did not object to the discontinuance of the proceedings and the Court recorded the discontinuance in an order of 10 April 1961. After the negotiations broke down, Belgium filed a new application on 19 June 1962, this time seeking reparation for the ‘damage suffered by Belgian nationals, individuals or legal persons, being shareholders of Barcelona Traction’.
II. THE 1964 JUDGMENT ON PRELIMINARY OBJECTIONS
With regard to this new claim, Spain raised four preliminary objections. The first
one was that the discontinuance affected the admissibility of the new application.
The discussion of this objection turned on whether there had been an understanding
between the parties that discontinuance involved a final waiver. The Court rejected
Spain’s objection mainly because it found that the various exchanges between the
parties were ‘wholly inconclusive’6 and that one could not reasonably ‘suppose that
on the eve of difficult negotiations, the success of which must be uncertain, there
could have been any intention on the Belgian side to forgo the advantage represented
by the possibility of renewed proceedings’"""
