from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import PropertyGraphIndex,Document


# 1. 定义知识图谱的三元组数据
triples = [
    ("Apple", "is a", "Fruit"),
    ("Apple", "has color", "Red"),
    ("Lion", "is a", "Animal"),
    ("Lion", "eats", "Meat"),
    ("Elephant", "is a", "Animal"),
    ("Elephant", "has size", "Large")
]

# 2. 将三元组数据转换为文本格式的文档
documents = []
for head, relation, tail in triples:
    text = f"{head} {relation} {tail}"
    documents.append(Document(text=text))

#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm
#
# # 4. 创建图索引
# index = VectorStoreIndex.from_documents(documents)
#
# # 5. 保存索引到本地（如果需要持久化）
# index.storage_context.persist(persist_dir="graph_index")
#
# query_engine = index.as_query_engine()
#
# # 6. 查询索引
# query = "What is Apple?"
# response = query_engine.query(query)
#
# # 输出查询结果
# print("Query response:", response)
#
# # 如果想从保存的索引加载
# storage_context = StorageContext.from_defaults(persist_dir="graph_index")
# loaded_index = load_index_from_storage(storage_context)
#
# # 使用加载的索引进行查询
# query_engine = loaded_index.as_query_engine()
# loaded_response = query_engine.query(query)
# print("Loaded query response:", loaded_response)


# # 创建几个示例文档
# doc1 = Document(text="小明吃了一个苹果。")
# doc2 = Document(text="小红喝了牛奶。")
# doc3 = Document(text="苹果是小明喜欢的水果。")
#
# # 将文档放入列表
# doc_list = [doc1, doc2, doc3]
#
# # 使用 from_documents 构建 PropertyGraphIndex
# graph_index = PropertyGraphIndex.from_documents(doc_list,use_async=False)
#
# # 打印或检查索引内容
# print(graph_index)


