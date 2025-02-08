from llama_index.core import Document,SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import pprint

llm = Ollama(model='qwen2.5')
embedded_model = \
OllamaEmbedding(model_name="herald/dmeta-embedding-zh",
embed_batch_size=50)
docs = SimpleDirectoryReader(input_files=["./data/unix_standard.txt"]).load_data()

#构造一个数据摄取管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
        TitleExtractor(llm=llm, show_progress=False)
    ]
)

#运行这个数据摄取管道
nodes = pipeline.run(documents=docs)
# print(nodes)

#=================================自定义转化器===============================
from llama_index.core.schema import TransformComponent
import re

#定义一个做数据清理的转换器
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9，。？！“”‘’；：【】《》（）\[\]\"\'\.\,\?\!\:\;\(\)\n\r]", "", node.text)
        return nodes


pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
        TextCleaner(),   #插入自定义转换器
        TitleExtractor(llm=llm, show_progress=False)
    ]
)
nodes = pipeline.run(documents=docs)
# print(nodes)
#
# =============================================================================
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery


#构造一个向量存储对象用于存储最后输出的Node对象
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(name="pipeline")
vector_store = ChromaVectorStore(chroma_collection=collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
        TitleExtractor(llm=llm, show_progress=False),
        embedded_model   #提供一个嵌入模型用于生成向量
    ],
    vector_store = vector_store   #提供一个向量存储对象，用于存储最后的Node对象
)
nodes = pipeline.run(documents=docs)

#==============================================================================
# 获取所有存储的嵌入向量（按ID）
results = collection.get()
print("------------------------------------------------------------")
print(results)
# 输出查看嵌入向量数据
for id, result in enumerate(results["documents"], start=1):
    print(f"id: {id}  result: ", result)

# print(f"Embedding: {results['embeddings'][0]}")

print("------------------------------------------------------------")
#==============================================================================


#用输入问题做语义检索
results  = vector_store.query(VectorStoreQuery(
                            query_str='文心一言是什么？',
                            similarity_top_k=3))
pprint.pprint(results.nodes)

