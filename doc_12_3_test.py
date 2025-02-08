from multiprocessing.managers import BaseManager

def test_query_index(msg):
    manager = BaseManager(('192.168.97.215', 5602), b'password')
    manager.register('query_index')
    manager.connect()
    response = manager.query_index(msg)._getvalue()
    print(response)

def test_client(doc_file_path, doc_id):
    # 连接到服务端
    manager = BaseManager(('192.168.97.215', 5602), b'password')
    manager.register('insert_into_index')  # 注册插入方法
    manager.register('get_documents_list')  # 注册获取文档列表方法
    manager.connect()

    # 插入文档
    # doc_file_path = "./data/beijing.txt"  # 文档路径
    # doc_id = "doc_001"  # 文档 ID
    manager.insert_into_index(doc_file_path, doc_id)  # 调用插入方法
    print(f"Document {doc_id} inserted successfully.")

    # 获取文档列表
    documents_list = manager.get_documents_list()._getvalue()  # 调用获取文档列表方法
    print("Stored Documents:")
    for doc in documents_list:
        print(f"ID: {doc['id']}, Text: {doc['text']}")

if __name__ == "__main__":
    # test_client("./data/nanjing.txt", "doc_002")
    # test_client("./data/shanghai.txt","doc_003")
    print("-----------------------------------------------------------")
    test_query_index("介绍一下上海市？")


