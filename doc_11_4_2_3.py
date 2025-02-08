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
import os
import pickle


#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = \
OllamaEmbedding(model_name="nomic-embed-text")

url = ['https://qwenlm.github.io/zh/blog/qwen1.5/']

#此处更改默认的摘要Prompt为中文
DEFAULT_SUMMARY_QUERY_STR = """\
尽可能结合上下文，用中文详细介绍表格内容。\
这个表格是关于什么的？给出一个摘要说明（想象你正在为这个表格添加一个新的标题和摘要），\
如果提供了上下文，请输出真实/现有的表格标题/说明。\
如果提供了上下文，请输出真实/现有的表格ID。\
"""

nodes_save_path = "data/raw_nodes.pkl"

#加载网页到docs变量
web_loader = SimpleWebPageReader()
docs = web_loader.load_data(url)
print(f"docs:  {docs}")

#分割成Node，并持久化存储
node_parser = UnstructuredElementNodeParser(
            summary_query_str=DEFAULT_SUMMARY_QUERY_STR)

if nodes_save_path is None or not os.path.exists (nodes_save_path):
    simple_doc = Document(text="This is a simple document for testing.")
    raw_nodes = node_parser.get_nodes_from_documents([simple_doc])
    pickle.dump(raw_nodes, open(nodes_save_path, "wb"))
else:
    raw_nodes = pickle.load(open(nodes_save_path, "rb"))
