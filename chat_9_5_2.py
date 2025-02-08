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
from llama_index.core.agent import AgentRunner, ReActAgentWorker
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

citys_dict = {
    '北京市': 'beijing',
    '南京市': 'nanjing',
    '广州市': 'guangzhou',
    '上海市': 'shanghai',
    '深圳市': 'shenzhen'
}
# 具体的城市查询函数
def query_beijing(all_sentence:str, query: str) -> str:
    """查询北京市的相关信息"""
    print(f"query:{query},sentence:{all_sentence}")
    if "人口" in query:
        return "北京市的人口约为 2171 万。"
    elif "面积" in query:
        return "北京市的面积约为 16410.54 平方公里。"
    elif "GDP" in query:
        return "北京市的 GDP 约为 4.03 万亿元。"
    else:
        return "北京市是中国的首都，政治、文化、国际交往和科技创新中心。"

beijing = FunctionTool.from_defaults(fn=query_beijing)

def query_shanghai(query: str) -> str:
    """查询上海市的相关信息"""
    if "人口" in query:
        return "上海市的人口约为 2487 万。"
    elif "面积" in query:
        return "上海市的面积约为 6340.5 平方公里。"
    elif "GDP" in query:
        return "上海市的 GDP 约为 4.32 万亿元。"
    else:
        return "上海市是中国的经济、金融、贸易和航运中心。"

shanghai = FunctionTool.from_defaults(fn=query_shanghai)

def query_guangzhou(query: str) -> str:
    """查询广州市的相关信息"""
    if "人口" in query:
        return "广州市的人口约为 1868 万。"
    elif "面积" in query:
        return "广州市的面积约为 7434.4 平方公里。"
    elif "GDP" in query:
        return "广州市的 GDP 约为 2.82 万亿元。"
    else:
        return "广州市是广东省的省会，华南地区的经济、文化和交通中心。"

guangzhou = FunctionTool.from_defaults(fn=query_guangzhou)
def query_nanjing(query: str) -> str:
    """查询南京市的相关信息"""
    if "人口" in query:
        return "南京市的人口约为 850 万。"
    elif "面积" in query:
        return "南京市的面积约为 6587 平方公里。"
    elif "GDP" in query:
        return "南京市的 GDP 约为 1.48 万亿元。"
    else:
        return "南京市是江苏省的省会，历史文化名城。"

nanjing = FunctionTool.from_defaults(fn=query_nanjing)

def query_shenzhen(query: str) -> str:
    """查询深圳市的相关信息"""
    if "人口" in query:
        return "深圳市的人口约为 1756 万。"
    elif "面积" in query:
        return "深圳市的面积约为 1997.47 平方公里。"
    elif "GDP" in query:
        return "深圳市的 GDP 约为 3.24 万亿元。"
    else:
        return "深圳市是中国的经济特区，科技创新中心。"

shenzhen = FunctionTool.from_defaults(fn=query_shenzhen)

# 创建城市查询工具
def create_city_tool(name: str):
    '''根据城市名构造对应的查询引擎，并将其包装成查询引擎工具'''
    if name == "北京市":
        query_fn = query_beijing
    elif name == "上海市":
        query_fn = query_shanghai
    elif name == "广州市":
        query_fn = query_guangzhou
    elif name == "南京市":
        query_fn = query_nanjing
    elif name == "深圳市":
        query_fn = query_shenzhen
    else:
        raise ValueError(f"未知的城市名称: {name}")

    return FunctionTool(
        fn=query_fn,
        metadata=ToolMetadata(
            name=f"query_{citys_dict[name]}",  # 工具名称，例如 query_beijing
            description=f"查询 {name} 的相关信息"  # 工具描述
        )
    )

# # 先构造一组工具，再开发一个Agent，也可以直接开发Agent
# query_engine_tools = []
# for city in citys_dict.keys():
#     query_engine_tools.append(create_city_tool(city))

query_engine_tools = [beijing, shenzhen, shanghai,nanjing,guangzhou]
# 开发Agent
openai_step_engine = ReActAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,  # 使用 Ollama 的 qwen2.5 模型
    verbose=True
)
agent = AgentRunner(openai_step_engine)

# 分步运行Agent：并要求人类给出指令
task_message = None
history = [
    {"role": "user", "content": "北京的人口多少"},
    {"role": "assistant", "content": "北京市的人口约为 2171 万。"}
]
while task_message != "exit":
    task_message = input(">> 你: ")
    if task_message == "exit":
        break

    # 根据输入问题构造任务
    task = agent.create_task(task_message)

    response = None
    step_output = None
    message = None

    # 任务执行过程中允许人类反馈信息
    # 如果message为exit，那么任务被取消，并退出
    # 如果任务步骤返回is_last=True，那么任务正常退出
    while message != "exit" and (not step_output or not step_output.is_last):

        # 执行下一步任务
        if message is None or message == "":
            step_output = agent.run_step(task.task_id)
        else:
            # 允许把人类反馈信息传入中间的任务步骤
            step_output = agent.run_step(task.task_id, input=message)

        # 如果任务没结束，那么允许用户输入
        if not step_output.is_last:
            message = input(">>  请补充任务反馈信息（留空继续，exit退出）: ")

    # 任务正常退出
    if step_output.is_last:
        print(">> 任务运行完成。")
        response = agent.finalize_response(task.task_id)
        print(f"Final Answer: {str(response)}")

    # 任务被取消
    elif not step_output.is_last:
        print(">> 任务未完成，被丢弃。")

