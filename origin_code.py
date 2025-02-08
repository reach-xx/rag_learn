import ollama, chromadb

# 引入自定义模块
from load import loadtext, getconfig
from splitter import split_text_by_sentences

# # 向量模型
embedmodel = getconfig()["embedmodel"]

# 向量库
chroma = chromadb.HttpClient(host="localhost", port=8000)
try:
    chroma.delete_collection(name="ragdb")
except:
    print("ragdb database not exist!!!")
collection = chroma.get_or_create_collection(name="ragdb")

# 读取文档列表，依次处理
with open('docs.txt') as f:
    lines = f.readlines()
    for filename in lines:
        # 加载文档内容
        text = loadtext(filename)

        # 把文档分割成知识块
        chunks = split_text_by_sentences(source_text=text,
                                         sentences_per_chunk=8,
                                         overlap=0)

        # 对知识块依次处理
        for index, chunk in enumerate(chunks):
            # 借助基于Ollama部署的本地嵌入模型生成向量
            print(f"##########################################index:{index}    chunk: {chunk}")
            embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']

            # 存储到向量库Chroma中，注意这里的参数
            collection.add([filename + str(index)], [embed], documents=[chunk], metadatas={"source": filename})
            print(f"----------------------------------------------------------------------------{filename}")

print("----------------------------------------finish-------------------------------------")
# if __name__ == "__main__":
#     while True:
#         query = input("Enter your query: ")
#         if query.lower() == 'quit':
#             break
#         else:
#             #从向量库Chroma中查询与向量相似的知识块
#             results = \
# collection.query(query_embeddings=[ollama.embeddings(model=embedmodel, prompt=query)['embedding']], n_results=3)
#
#             print(f"**********results: {results}")
#
#             #打印文档内容（Chunk）
#             for result in results["documents"][0]:
#                 print(result)
      #嵌入式的使用方式访问chromadb数据库
#     client = chromadb.PersistentClient(path="./chroma_db")
#     print(client.heartbeat())
      #采用C/S的方式访问chromadb数据库
    # chromadb_client = chromadb.HttpClient(host='localhost', port=8000)
    # print(chromadb_client.heartbeat())



