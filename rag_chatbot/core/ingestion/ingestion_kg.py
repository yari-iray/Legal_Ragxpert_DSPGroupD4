import pathlib
from uuid import uuid4
import PyPDF2
import networkx as nx
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import matplotlib.pyplot as plt
from spacy.tokens import Doc
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

class DocumentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        # Increase max_length to handle large texts
        self.nlp.max_length = 3_000_000  
        
        self.logger = logging.Logger("doc-analyzer")
        
        self.ner_model_name = "dslim/bert-base-NER"
        self.ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(self.ner_model_name)
        
    def read_pdf(self, pdf_path: str) -> str:      
        p = pathlib.Path().resolve()
          
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            
        return text
    
    def preprocess_text(self, text: str) -> Doc:
        if len(text) <= self.nlp.max_length:
            return self.nlp(text)

        chunks = [text[i:i+500_000] for i in range(0, len(text), 500_000)]
        docs = [self.nlp(chunk) for chunk in chunks]
        return Doc.from_docs(docs)       

    
    def _get_entity_type(self, ent) -> str:
        """Classify entity into FLINT categories: Act, Fact, Duty, Agent."""
        category_map = {
            "ORG": "Agent", "PERSON": "Agent", "GPE": "Agent",
            "EVENT": "Act", "LAW": "Act", "NORP": "Fact", "MONEY": "Fact",
            "DATE": "Fact", "TIME": "Fact", "CARDINAL": "Fact"
        }
        return category_map.get(ent.label_, "Misc")
    
    def _normalize_entity_name(self, ent) -> str:
        """Generate more meaningful names by preserving key context words."""
        
        words = [token.text for token in ent if token.pos_ in {"NOUN", "PROPN", "ADJ"}]
        
        return " ".join(words).title() if words else ent.text.title()
    
    def _extract_description(self, sentence: str, entity_text: str, max_length: int = 30000) -> str:
        desc = sentence.strip()
        
        if len(desc) > max_length:
            cutoff = desc.rfind('.', 0, max_length)
            if cutoff == -1:
                cutoff = desc.rfind(' ', 0, max_length)
            desc = desc[:cutoff] + "..." if cutoff != -1 else desc[:max_length] + "..."
            
        return desc
    
    def extract_entities(self, doc: Doc) -> dict[str, dict[str, str]]:
        entities: dict[str, dict[str, str]] = {}
        seen_descriptions = set()
        
        for ent in doc.ents:
            if len(ent.text.strip()) <= 1:
                continue
            
            name = self._normalize_entity_name(ent)
            if not name:
                continue 
            
            name = name.casefold()
            
            entity_type = self._get_entity_type(ent)
            description = self._extract_description(ent.sent.text, ent.text)
            if description and description not in seen_descriptions:
                entities[name] = {
                    'name': name,
                    'type': entity_type,
                    'description': description,
                    'original_text': ent.text
                }
                seen_descriptions.add(description)
                        
        return entities
    
    def extract_relationships(self, doc: Doc, entities: Dict[str, dict]) -> List[Tuple[str, str]]:
        relationships = []
        entity_pairs = self._get_entity_pairs(entities)

        for sent in doc.sents:
            sent_text = sent.text.casefold()
            present_pairs = [(e1, e2) for e1, e2 in entity_pairs if e1 in sent_text and e2 in sent_text]
            relationships.extend(present_pairs)
        return relationships
    
    def _get_entity_pairs(self, entities: Dict[str, dict]) -> Set[Tuple[str, str]]:
        entity_list = list(entities.keys())
        
        return {(e1.casefold(), e2.casefold()) for i, e1 in enumerate(entity_list) for e2 in entity_list[i+1:]}
    
    
    
    def create_graph(self, entities: dict[str, dict[str, str]], relationships: List[Tuple[str, str]]) -> nx.Graph:
        entity_graph = nx.Graph()
        node_ids = {name: str(uuid4()) for name in entities.keys()}
        
        for name, props in entities.items():
            entity_graph.add_node(
                node_ids[name],
                id=node_ids[name],
                label=name,  # Store name as separate property
                type=props['type'],
                description=props['description'],
                original_text=props['original_text']
            )
            
        for e1, e2 in relationships:
            if e1 in node_ids.keys() and e2 in node_ids.keys():  # Safety check
                entity_graph.add_edge(
                    node_ids[e1], 
                    node_ids[e2],
                    relationship_type="RELATED_TO",  # Explicit type
                    weight=1.0  # Default weight
                )
            
        return entity_graph
        
        
        
        
    def analyze_document(self, pdf_path: str):
        text = self.read_pdf(pdf_path)
        doc = self.preprocess_text(text)
        
        entities = self.extract_entities(doc)
        relationships = self.extract_relationships(doc, entities)
        
        graph = self.create_graph(entities, relationships)
        
        return graph, entities, relationships
    

def visualize_graph(G: nx.Graph):
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    node_colors = []
    node_types = nx.get_node_attributes(G, 'type')
    unique_types = set(node_types.values())
    color_map = {
        "Agent": "blue", "Act": "red", "Fact": "green", "Duty": "yellow", "Misc": "gray"
    }
    for node in G.nodes():
        node_colors.append(color_map.get(G.nodes[node]['type'], "gray"))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, label=node_type,
                                    markersize=10)
                        for node_type, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Document Entity Graph (FLINT-Based)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def run_main():
    analyzer = DocumentAnalyzer()
    pdf_path = "constitution.pdf"  # Replace with your PDF path
    
    graph, entities, relationships = analyzer.analyze_document(pdf_path)
    
    # visualize_graph(graph)
    
    nx.write_graphml(graph, "constitution_graph.graphml")
    
if __name__ == "__main__":
    run_main()
    
    
    
    
"""
OLD CODE:

    def extract_relationships(self, doc: Doc, entities: Dict[str, dict]) -> List[Tuple[str, str]]:
        relationships = []
        lowercase_pairs = self._get_entity_pairs(entities)        
        
        entity_to_pairs = defaultdict(list)
        all_entities = set()
        
        for lc_e1, lc_e2 in lowercase_pairs:
            entity_to_pairs[lc_e1].append((lc_e1, lc_e2))
            entity_to_pairs[lc_e2].append((lc_e1, lc_e2))
            all_entities.update({lc_e1, lc_e2})
        
        # Precompute all lowercase entities for fast membership test
        all_entities = frozenset(all_entities)
        
        for sent in doc.sents:  # Removed tqdm for production use
            # Fast sentence processing
            words = sent.text.casefold().split()
            unique_words = frozenset(words)
            
            # Find relevant entities using set intersection
            present_entities = unique_words & all_entities
            if not present_entities:
                continue
            
            # Efficient pair checking with early exits
            found_pairs = set()
            for entity in present_entities:
                for pair in entity_to_pairs[entity]:
                    other = pair[0] if pair[1] == entity else pair[1]
                    if other in present_entities:
                        found_pairs.add(pair)
            
            relationships.extend(found_pairs)
        
        return relationships
"""