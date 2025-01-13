from setting import RAGSettings
from langchain_community.chains.graph_qa.base import GraphQAChain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain_ollama.llms import OllamaLLM
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

check_components_are_running = True


SYSTEM_PROMPT_RAG_EN = """You are from now on an expert on movies. 
The way you do this is by generating Cyper query language statements to query the database.
This should be your only ouput"""

setting = RAGSettings()
host = "localhost"

ollama_llm = OllamaLLM(model="llama3.2:1b",
                # system_prompt=SYSTEM_PROMPT_RAG_EN,
                base_url=f"http://{host}:{setting.ollama.port}",
                temperature=setting.ollama.temperature,
                context_window=setting.ollama.context_window,
                request_timeout=setting.ollama.request_timeout,
)

# Initialize graph and LLM
graph = Neo4jGraph(url=f"bolt://{host}:7687", username="neo4j", password="password", database="neo4j")

if check_components_are_running:
    print(ollama_llm.invoke("hello"))
    print(graph.query("MATCH (n) RETURN n LIMIT 1"))

# v_i = Neo4jVector.from_existing_graph(
#     OllamaEmbeddings(),
    
# )

chain = GraphCypherQAChain.from_llm(
    llm = ollama_llm,
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True
)

# Create a GraphQA chain
# chain = GraphQAChain.from_llm(ollama_llm, graph= graph)

# Query the system
response = chain.invoke("What movies were released in the year 1999?")
print("Response from the LLM: ")
print(response)