from llama_index.core import Document,SimpleDirectoryReader,Settings,load_index_from_storage
# from llama_index.core.node_parser import SentenceSplitter,SentenceWindowNodeParser
# from llama_index.core.extractors import TitleExtractor
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
from llama_index.core import PropertyGraphIndex
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

#指定知识图谱的存储，这里使用内存存储
property_graph_store = SimplePropertyGraphStore()
storage_context = StorageContext.from_defaults(property_graph_store=property_graph_store)

#构造知识图谱索引（这里进行了本地化存储）
if not os.path.exists(f"./storage/graph_store"):
    index = PropertyGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        use_async = False
    )
    index.storage_context.persist(persist_dir="./storage/graph_store")
else:
    print('Loading graph index...')
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./storage/graph_store")
    )

query_engine = index.as_query_engine()
retriever = index.as_retriever(similarity_top_k = 1)
nodes = (retriever.retrieve("介绍一下西京的城市信息吧"))
print("----------------------------------")
pprint.pprint(nodes)
print("----------------------------------")

#构造查询引擎
query_engine = index.as_query_engine(
    include_text=True, similarity_top_k=2
)

#查看构造的知识图谱索引（这里打印了保存的三元组）
graph = index.property_graph_store.graph
pprint.pprint(graph.triplets)

response = query_engine.query(
    "介绍一下西京的城市信息吧",
)
print(f"Response: {response}")
