import os
import pickle
import chromadb
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.core import Settings,SimpleDirectoryReader, VectorStoreIndex,StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

index = None
stored_docs = {}

# 确保线程安全
lock = Lock()

# 索引持久化存储的位置
index_name = "./storage/saved_index"
pkl_name = "./storage/stored_documents.pkl"

# 索引服务端口
SERVER_PORT = 5602


# 初始化索引
def initialize_index():
    global index, stored_docs

    # 构造向量存储
    chroma = chromadb.HttpClient(host="localhost", port=8080)
    collection = chroma.get_or_create_collection(name="chat_docs_collection")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # 注意使用with lock进行互斥的索引访问
    with lock:

        # 如果已经存在持久化存储的数据，那么加载
        if os.path.exists(index_name):
            storage_context = StorageContext.from_defaults(
                persist_dir=index_name,
                vector_store=vector_store)
            index = load_index_from_storage(storage_context=storage_context)
        else:
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store)

            # 首次构造空的索引，后面再插入
            index = VectorStoreIndex([], storage_context=storage_context)
            index.storage_context.persist(persist_dir=index_name)

        # 将已经上传的文档信息从存储的文档中读取到内存
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                stored_docs = pickle.load(f)

# 定义查询索引的方法
def query_index(query_text):
    """Query the global index."""
    global index
    response = index.as_query_engine().query(query_text)
    print(f"response: {response}")
    return response


# 在已有的索引中插入新的文档对象
def insert_into_index(doc_file_path, doc_id=None):
    """在已有的索引中插入新的文档对象."""
    global index, stored_docs
    document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    if doc_id is not None:
        document.doc_id = doc_id

    # 使用with lock实现共享对象的互斥（顺序）访问
    with lock:
        index.insert(document)
        index.storage_context.persist(persist_dir=index_name)

        # 这里简化使用，只读取前200个字符
        stored_docs[document.doc_id] = document.text[0:200]  # only take the first 200 chars

        with open(pkl_name, "wb") as f:
            pickle.dump(stored_docs, f)

    return


def get_documents_list():
    """Get the list of currently stored documents."""
    global stored_doc
    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})

    return documents_list


if __name__ == "__main__":
    # 初始化索引
    print("initializing index...")
    initialize_index()

    # 构造manager
    print(f'Create server on port {SERVER_PORT}...')
    manager = BaseManager(('', SERVER_PORT), b'123456')

    # 注册函数
    print("registering functions...")
    manager.register('query_index', query_index)
    manager.register('insert_into_index', insert_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    # 启动server
    print("server started...")
    server.serve_forever()