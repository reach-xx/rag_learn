from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    text
)
from sqlalchemy.orm import sessionmaker
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core import Document,SimpleDirectoryReader,Settings,load_index_from_storage,get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core.tools import ToolMetadata
import pprint
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index.core import PropertyGraphIndex,TreeIndex,KeywordTableIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import VectorStoreIndex,SummaryIndex,SimpleKeywordTableIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline.components.input import InputComponent
from llama_index.core.tools import BaseTool
from typing import Any, Callable, Optional, Sequence,List
from llama_index.core.llms import ChatMessage
from openai.types.chat import ChatCompletionMessageToolCall
# from llama_index.core.agent.workflow.workflow_events import ToolCall
import json


langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key="pk-lf-b7bffd3d-1b28-4799-8147-8e8f277770c5",
    secret_key="sk-lf-c83f7342-bdd1-4b06-9e3d-e2f814ee93a3",
    host="https://cloud.langfuse.com"
)
Settings.callback_manager = CallbackManager([langfuse_callback_handler])

#工具：搜索天气情况
def search_weather(query: str) -> str:
    """用于搜索天气情况"""
    # Perform search logic here
    search_results = f"明天晴转多云，最高温度30℃，最低温度23℃。天气炎热，注意防晒哦。"
    return search_results

tool_search = FunctionTool.from_defaults(fn=search_weather)

#工具：发送电子邮件
def send_email(subject: str, recipient: str, message: str) -> None:
    """用于发送电子邮件"""
    # Send email logic here
    print(f"邮件已发送至 {recipient}，主题为 {subject}，内容为 {message}")

tool_send_mail = FunctionTool.from_defaults(fn=send_email)

#工具：查询客户信息
def query_customer(phone: str) -> str:
    """用于查询客户信息"""
    # Perform creation logic here
    result = f"该客户信息为:\n姓名: 张三\n电话: {phone}\n地址: 北京市海淀区"
    return result

tool_generate = FunctionTool.from_defaults(fn=query_customer)


# 定义一个Ollama的Agent
class MyOllamaAgent:

    # 初始化参数
    # tools: Sequence[BaseTool] = []，工具列表
    # llm: Ollama = Ollama(model="qwen2.5")，Ollama的大模型
    # chat_history: List[ChatMessage] = []，聊天历史
    def __init__(
            self,
            tools: Sequence[BaseTool] = [],
            llm: Ollama = Ollama(model="qwen2.5"),
            chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    # 重置聊天历史
    def reset(self) -> None:
        self._chat_history = []

    # # 定义聊天接口
    # def chat(self, message: str) -> str:
    #     chat_history = self._chat_history
    #     chat_history.append(ChatMessage(role="user", content=message))
    #
    #     # 传入工具
    #     tools = [
    #         tool.metadata.to_openai_tool() for _, tool in self._tools.items()
    #     ]
    #     ai_message = self._llm.chat(chat_history, tools=tools).message
    #     additional_kwargs = ai_message.additional_kwargs
    #     chat_history.append(ai_message)
    #
    #     # 获取工具调用的要求
    #     tool_calls = additional_kwargs.get("tool_calls", None)
    #
    #     # 如果调用工具，那么依次调用
    #     if tool_calls is not None:
    #         for tool_call in tool_calls:
    #             # 调用函数
    #             function_message = self._call_function(tool_call)
    #             chat_history.append(function_message)
    #
    #             # 继续对话
    #             ai_message = self._llm.chat(chat_history).message
    #             chat_history.append(ai_message)
    #
    #     return ai_message.content
    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))

        # 传入工具
        tools = [tool.metadata.to_openai_tool() for _, tool in self._tools.items()]
        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)
        print(f"ai_message: {ai_message},  additional_kwargs: {additional_kwargs} ")

        # 获取工具调用的要求
        tool_calls = additional_kwargs.get("tool_calls", None)

        # 如果调用工具，那么依次调用
        if tool_calls is not None:
            for tool_call in tool_calls:
                # 调用函数
                function_message = self._call_function(tool_call)
                print("tool_call: ", tool_call)
                print("function_message: ", function_message)
                chat_history.append(function_message)

                # 继续对话
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    # 调用函数
    # def _call_function(
    #         self, tool_call: ChatCompletionMessageToolCall
    # ) -> ChatMessage:
    #     id_ = tool_call.id
    #     function_call = tool_call.function
    #     tool = self._tools[function_call.name]
    #     output = tool(**json.loads(function_call.arguments))
    #     return ChatMessage(
    #         name=function_call.name,
    #         content=str(output),
    #         role="tool",
    #         additional_kwargs={
    #             "tool_call_id": id_,
    #             "name": function_call.name,
    #         },
    #     )
# 调用函数
    def _call_function(self, tool_call: Any) -> ChatMessage:
        """
        根据 Ollama 的工具调用对象结构解析并调用工具。
        """
        print(f"tool_call: {tool_call}")
        function = tool_call.function
        # 假设 tool_call 是一个包含工具名称和参数的字典
        function_name = function.name  # 获取工具名称
        print(f"function_name: {function_name}")
        function_args = function.arguments  # 获取工具参数
        print(f"function_name: {function_name}   function_args: {function_args} ")

        # 获取工具实例
        tool = self._tools.get(function_name)
        if not tool:
            raise ValueError(f"工具 '{function_name}' 未找到")

        # 调用工具
        output = tool(**function_args)

        # 返回工具调用结果
        return ChatMessage(
            name=function_name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": "unknown",  # 如果 Ollama 不支持 tool_call_id，可以设置为占位符
                "name": function_name,
            },
        )

# 初始化Agent
agent = MyOllamaAgent(tools=[tool_search, tool_send_mail, tool_generate])

while True:
    user_input = input("请输入您的消息：")
    if user_input.lower() == "quit":
        break
    response = agent.chat(user_input)
    print(response)
    langfuse_callback_handler.flush()