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

embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

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

tools = [tool_search,tool_send_mail,tool_generate]

from llama_index.core.objects import ObjectIndex
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.query_engine import RetrieverQueryEngine

#构造对象索引，在底层使用向量检索
obj_index = ObjectIndex.from_objects(
    tools,
    index_cls=VectorStoreIndex,
)

tool_retriever=obj_index.as_retriever(similarity_top_k=2)
print(tool_retriever)

# #开发Agent，注意提供tool_retriever(工具检索器)，而不是工具集
# agent = OpenAIAgent.from_tools(
#     tool_retriever=obj_index.as_retriever(similarity_top_k=2),
#     verbose=True
# )
#
# #测试：使用agent_worker对象检索工具集，查看结果
# tools = agent.agent_worker.get_tools('发送电子邮件')
# for tool in tools:
#     print(f'Tool name: {tool.metadata.name}')
# print(tools)