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
def search_weather(address: str, date:str) -> str:
    """用于搜索天气情况"""
    # Perform search logic here
    search_results = f"今天晴转多云，最高温度35℃，最低温度28℃。天气炎热，注意防晒哦。"
    print(f"address: {address}  date: {date}")
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

#工具：巡课
def query_course(campus: str, building:str, floor: str, room:str) -> str:
    """用于查询学校上课情况信息"""
    # Perform creation logic here
    result = f"查询的地址为: campus: {campus}, building: {building} floor: {floor}, room: {room}"
    return result
tool_course = FunctionTool.from_defaults(fn=query_course)

from llama_index.core.agent import ReActAgent

tools = [tool_search,tool_send_mail, tool_generate,tool_course]
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

#构造一个任务
task = agent.create_task("帮我查下五山校区博学楼的上课情况？")

#运行这个任务
print('\n--------------')
step_output = agent.run_step(task.task_id)
pprint.pprint(step_output.__dict__)

#循环，直到is_last = True
while not step_output.is_last:
    print('\n--------------')
    step_output = agent.run_step(task.task_id)
    pprint.pprint(step_output.__dict__)

#最后输出结果
print('\nFinal response:')
response = agent.finalize_response(task.task_id)
print(str(response))

print("------------------------------------------------")
steps = agent.get_completed_steps(task.task_id)
for i,step in enumerate(steps):
    print(f'\nStep {i+1}:')
    print(step.__dict__)