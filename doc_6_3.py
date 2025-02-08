from llama_index.core import Document,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import pprint
from llama_index.core.schema import MetadataMode
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core import VectorStoreIndex,StorageContext

#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

#加载与解析文档，这里直接构造
docs = [
Document(text="智家机器人是一种人工智能家居软件，让您的家变得更智能，让您轻松地掌控生活的方方面面。",metadata={"title":"智家机器人"},doc_id="doc1"),
Document(text="速达飞行者是一种飞行汽车，能够让您在城市中自由翱翔，体验全新的出行方式。",metadata={"title":"速达飞行者"},doc_id="doc2")]

splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
nodes = splitter.get_nodes_from_documents(docs)

#构造一个向量存储对象用于存储最后输出的Node对象
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(name="pipeline")
vector_store = ChromaVectorStore(chroma_collection=collection)

# nodes = embedded_model(nodes)
# vector_store.add(nodes)
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("什么是速达飞行者")
# print(response)

#================================从文档中直接构造向量存储索引对象================================
mySplitter = SentenceWindowNodeParser(window_size=2)
myExtractor = TitleExtractor()
vector_index = VectorStoreIndex.from_documents(docs,storage_context=storage_context, transformations=[mySplitter, myExtractor])
query_engine = vector_index.as_query_engine()
response = query_engine.query("请解释什么是区块链技术")
pprint.pprint(response)

