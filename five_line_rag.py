#经典的5行代码的RAG应用
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
#加载文档
documents = SimpleDirectoryReader("./data").load_data()

#构造向量存储索引
index = VectorStoreIndex.from_documents(documents)

#构造查询引擎
query_engine = index.as_query_engine()

#对查询引擎提问
response = query_engine.query('这里放入data目录中知识相关的问题')

#输出答案
print(response)