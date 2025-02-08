from llama_index.core.llms import ChatMessage
# from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
llm = Ollama(model='qwen2.5')
#测试complete接口
# llm = OpenAI(model='gpt-3.5-turbo-1106')
resp = llm.complete("白居易是")
print(resp)

#测试chat接口
messages = [
    ChatMessage(
        role="system", content="你是一个聪明的AI助手"
    ),
    ChatMessage(role="user", content="你叫什么名字？"),
]
resp = llm.chat(messages)
print(resp)