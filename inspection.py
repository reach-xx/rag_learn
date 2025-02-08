import ollama, sys, chromadb
from load import getconfig

# 嵌入模型与大模型
embedmodel = getconfig()["embedmodel"]
llmmodel = getconfig()["mainmodel"]

while True:
    user_input = input("Enter your query: ")
    if user_input.lower() == 'quit':
        break
    else:
        # 生成Prompt
        modelquery = f"""
                你是一个助手，根据用户输入提取意图和参数。返回 JSON 格式输出。
                示例：
                用户输入："帮我查询明天的天气"
                输出：{{"action": "query_weather", "parameters": {{"date": "2025-01-07", "location": "default"}}}}
                
                用户输入："帮我生成一份销售报告"
                输出：{{"action": "generate_report", "parameters": {{"type": "sales"}}}}
                
                用户输入："{user_input}"
                输出：
            """
        # 交给大模型进行生成
        stream = ollama.generate(model=llmmodel, prompt=modelquery, stream=True)
        # 流式输出生成的结果
        for chunk in stream:
            if chunk["response"]:
                print(chunk['response'], end='', flush=True)
