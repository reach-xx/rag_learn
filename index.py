import ollama, sys, chromadb
from load import getconfig

#嵌入模型与大模型
embedmodel = getconfig()["embedmodel"]
llmmodel = getconfig()["mainmodel"]

#向量库
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection("ragdb")

while True:
    query = input("Enter your query: ")
    if query.lower() == 'quit':
        break
    else:
        #生成查询向量
        queryembed = ollama.embeddings(model=embedmodel,
                                         prompt=query)['embedding']

        #用查询向量检索上下文
        relevantdocs = collection.query(query_embeddings=[queryembed],
                                         n_results=5)["documents"][0]
        docs = "\n\n".join(relevantdocs)

        #生成Prompt
        modelquery = f"""
        请基于以下的上下文回答问题，如果上下文中不包含足够的回答问题的信息，请回答'我暂时无法回答该问题'，不要编造。
    
        上下文： 
        ====
        {docs}
        ====
    
        我的问题是：{query}
        """

        #交给大模型进行生成
        stream = ollama.generate(model=llmmodel, prompt=modelquery, stream=True)
        #流式输出生成的结果
        for chunk in stream:
            if chunk["response"]:
                print(chunk['response'], end='', flush=True)

