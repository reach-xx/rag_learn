from llama_index.core import Document,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter
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

#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
Settings.embed_model=embedded_model

#docs
docs = [Document(text="百度文心一言是什么？文心一言是百度的大模型品牌。",metadata={"title":"百度文心一言的概念"},doc_id="doc1"),
        Document(text="什么是大模型？大模型是一种生成式推理AI模型。",metadata={"title":"大模型的概念"},doc_id="doc2")]
splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
nodes = splitter.get_nodes_from_documents(docs)

#生成嵌入向量
nodes = embedded_model(nodes)

#=============================SimpleVectorStore使用方式===============================
# print(nodes)
#存储到向量库中
simple_vectorstore = SimpleVectorStore()
simple_vectorstore.add(nodes)
# print(simple_vectorstore)

#查询
result = simple_vectorstore.query(
VectorStoreQuery(query_embedding=embedded_model.get_text_embedding('什么是语言模型'),similarity_top_k=1))
print(result)
simple_vectorstore.persist()
#=============================SimpleVectorStore使用方式===============================
#构造一个collection对象,此处使用Server模式下的Chroma向量库
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(name="vectorstore")
print("count: ", collection.count())

#构造向量存储对象
vector_store = ChromaVectorStore(chroma_collection=collection)
ids = vector_store.add(nodes)
print(f'{len(ids)} nodes ingested into vector store')
pprint.pprint(vector_store.__dict__)

#语义检索
result =\
vector_store.query(VectorStoreQuery(query_embedding=embedded_model.get_text_embedding('什么是语言模型'),similarity_top_k=2))
print(result)












