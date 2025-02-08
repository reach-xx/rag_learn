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
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex,StorageContext
import json
import ollama, sys, chromadb
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, CorrectnessEvaluator
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.retrievers.recursive_retriever import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.agent import ReActAgent
import os
import pickle


#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = \
OllamaEmbedding(model_name="nomic-embed-text")

chroma = chromadb.HttpClient(host="localhost", port=8080)
# chroma.delete_collection(name="ragdb1")
collection = chroma.get_or_create_collection(name="ragdb1", metadata={"hnsw:space": "cosine"})

# 创建针对某个文档的Agent
def create_file_agent(file, name):
    print(f'Starting to create tool agent for 【{name}】...\n')

    # ......省略构造文档对应的向量索引对象的过程......
    # vector_index = ...
    # 加载与读取文档
    reader = SimpleDirectoryReader(input_files=[f"{file}"])
    documents = reader.load_data()

    # 分割文档
    node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

    # 准备向量存储
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # 准备向量存储索引
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    # 构造查询引擎
    query_engine = vector_index.as_query_engine(similarity_top_k=3)

    # 构造摘要索引
    summary_index = SummaryIndex(nodes)
    summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize")

    # 将查询引擎“工具化”
    query_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=f'query_tool',
        description=f'Use if you want to query details about {name}')
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        name=f'summary_tool',
        description=f'Use ONLY IF you want to get a holistic summary of the documents. DO NOT USE if you want to query some details about {name}.')

    # 创建文档Agent
    file_agent = ReActAgent.from_tools([query_tool, summary_tool],
                                       verbose=True,
                                       system_prompt=f"""You are a specialized agent designed to answer queries about {name}.You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.DO NOT fabricate answer."""
                                       )
    return file_agent


##############################单个例子##########################
# agent = create_file_agent('./data/nanjing.txt', 'Nanjing')
# # agent.chat_repl()
# while True:
#     # 获取用户输入
#     user_input = input("You: ")
#
#     # 检查退出条件
#     if user_input.lower() in ["exit", "quit"]:
#         print("Goodbye!")
#         break
#
#     try:
#         # 调用 agent.chat() 处理输入
#         response = agent.chat(user_input)
#         print(f"Agent: {response}")
#     except Exception as e:
#         # 捕获并处理异常
#         print(f"An error occurred: {e}")
#         print("Please try again.")
##############################单个例子##########################

names = ['Nanjing','Shanghai','Beijing']
files = ['./data/nanjing.txt','./data/shanghai.txt','./data/beijing.txt']

#创建不同的文档Agent
print('===============================================\n')
print('Creating file agents for different documents...\n')
file_agents_dict = {}
for name, file in zip(names, files):
    file_agent = create_file_agent(file, name)
    file_agents_dict[name] = file_agent

print('===============================================\n')
print('Creating top level nodes from tool agents...\n')
index_nodes = []
query_engine_dict = {}

#给每个文档都构造一个索引Node，用于搜索
for name in names:
    doc_summary = f"这部分内容包含关于城市{name}的维基百科文章。如果您需要查找城市{name}的具体事实，请使用此索引。\n如果您想分析多个城市，请不要使用此索引。"
    node = IndexNode(
        index_id=name,
        text=doc_summary,
    )
    index_nodes.append(node)

    #把index_id与真正的Agent对应起来，用于在递归检索时查找
    #注意Agent也是一种查询引擎
    query_engine_dict[name] = file_agents_dict[name]

#构造一级索引与检索器
top_index = VectorStoreIndex(index_nodes)
top_retriever = top_index.as_retriever(similarity_top_k=1)

#构造递归检索器，从上面的top_retriever对象开始
#传入query_engine_dict变量，用于在递归检索时找到二级文档Agent
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": top_retriever},
    query_engine_dict=query_engine_dict,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
response = query_engine.query('北京市有哪些著名的旅游景点呢？')
print(response)



