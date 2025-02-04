from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from rdflib import Graph

from setting.setting import RAGSettings


def import_nodes(file_path: str, settings: RAGSettings | None = None):
    settings = settings or RAGSettings()

    auth_data = {'uri': settings.neo4j.url,
            'database': settings.neo4j.database,
            'user': settings.neo4j.username,
            'pwd': settings.neo4j.password}

    # Define your custom mappings & store config
    config = Neo4jStoreConfig(auth_data=auth_data,                        
        handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE, batching=True)

    neo4j_aura = Graph(store=Neo4jStore(config=config))
    # Calling the parse method will implictly open the store
    neo4j_aura.parse(file_path, format="ttl")
    neo4j_aura.close(True)
    
if __name__ == "__main__":
    path = "./ont_v2.ttl"
    import_nodes(path)