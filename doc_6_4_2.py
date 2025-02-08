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
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.objects.base import SimpleObjectNodeMapping, ObjectIndex

#构造一些不同类型的普通对象
obj1 = {"name": "小米","cpu": "骁龙","battery": "5000mAh","display": "6.67英寸"}
obj2 = ["iPhne", "小米", "华为", "三星"]
obj3 = (['A','B','C'],[100,200,300 ])
obj4 = "大模型是一种基于自然语言处理技术的生成式AI模型!"
objs= [obj1, obj2, obj3,obj4]

#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

# 从普通对象到Node对象的映射，即生成嵌入所需要的Node对象
obj_node_mapping = SimpleObjectNodeMapping.from_objects(objs)
nodes = obj_node_mapping.to_nodes(objs)

#构造一个向量存储对象用于存储最后输出的Node对象
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(name="pipeline")
vector_store = ChromaVectorStore(chroma_collection=collection)
# 构造对象索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
object_index = ObjectIndex(
    index=VectorStoreIndex(nodes=nodes, storage_context=storage_context),
    object_node_mapping=obj_node_mapping,
)

#构造一个检索器，测试检索结果
object_retriever = object_index.as_retriever(similarity_top_k=1)
results = object_retriever.retrieve("5000mAh")
print(f'results: {results}')
