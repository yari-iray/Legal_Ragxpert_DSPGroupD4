from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from rdflib import Graph

auth_data = {'uri': "bolt://localhost:7687",
         'database': "versioneight",
         'user': "neo4j",
         'pwd': "password"}

# Define your custom mappings & store config
config = Neo4jStoreConfig(auth_data=auth_data,
                      
handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE, batching=True)

file_path = './ont_v2.ttl'

# Create the RDF Graph, parse & ingest the data to Neo4j, and close the store(If the field batching is set to True in the Neo4jStoreConfig, remember to close the store to prevent the loss of any uncommitted records.)
neo4j_aura = Graph(store=Neo4jStore(config=config))
# Calling the parse method will implictly open the store
neo4j_aura.parse(file_path, format="ttl")
neo4j_aura.close(True)