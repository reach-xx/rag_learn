from llama_index.core import Document,SimpleDirectoryReader,Settings,load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import pprint
import os
from llama_index.core.schema import MetadataMode
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core import PropertyGraphIndex,TreeIndex,KeywordTableIndex
#
#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm
#
documents = SimpleDirectoryReader(
    input_files=["./data/graph.txt"],
).load_data()

print(documents)

splitter = SentenceSplitter(chunk_size=200, chunk_overlap=0)

nodes = splitter.get_nodes_from_documents(documents)

#============================TreeIndex#==============================
# pprint.pprint(nodes)
# index = TreeIndex(nodes)
# print(index.index_struct)
#============================TreeIndex#==============================

#============================
index = KeywordTableIndex(nodes)
print(index.index_struct)
query_engine = index.as_query_engine()
retriever = index.as_retriever(similarity_top_k = 1)
nodes = (retriever.retrieve("介绍一下西京的城市信息吧"))
print("----------------------------------")
pprint.pprint(nodes)
print("----------------------------------")
response = query_engine.query("介绍一下西京的城市信息吧")
print("response: ", response)
