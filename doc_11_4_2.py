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
import os


#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = \
OllamaEmbedding(model_name="nomic-embed-text")

docs = SimpleDirectoryReader(input_files=["./data/caffe.pdf"]).load_data()

#向量库
chroma = chromadb.HttpClient(host="localhost", port=8080)

def create_base_index():
    # 分割文档
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=0)
    nodes = splitter.get_nodes_from_documents(docs)

    # 设置每个Node的id为固定值
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"
    print("-----------------------------")
    print(f"nodes:  {nodes}")
    print("-----------------------------")
    collection = chroma.get_or_create_collection(name=f"crag")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    if not os.path.exists(f"./storage/vectorindex/crag"):
        print('Creating vector index...\n')
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model)
        vector_index.storage_context.persist(persist_dir=f"./storage/vectorindex/crag")
    else:
        print('Loading vector index...\n')
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/vectorindex/crag",
            vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)
    return vector_index, nodes

# 构造父Node
base_index, base_nodes = create_base_index()
# print(base_index)

def create_subnodes_index(base_nodes):
    # 构造两个不同粒度的分割器
    sub_chunk_sizes = [128, 256]
    sub_node_parsers = [SentenceSplitter(chunk_size=subsize, chunk_overlap=0) for subsize in sub_chunk_sizes]

    all_nodes = []
    # 对每一个父Node都进行分割
    for base_node in base_nodes:
        for n in sub_node_parsers:

            # 使用get_nodes_from_documents方法生成子Node
            sub_nodes = n.get_nodes_from_documents([base_node])
            for sn in sub_nodes:
                # 子Node是IndexNode类型的，并用父Node的id作为index_id
                indexnode_sn = \
                    IndexNode.from_text_node(sn, base_node.node_id)
                all_nodes.append(indexnode_sn)

        # 父Node也作为IndexNode对象放入all_nodes对象中
        all_nodes.append(IndexNode.from_text_node(base_node, base_node.node_id))

    # 构造子Node的向量索引
    collection = chroma.get_or_create_collection(name=f"crag-subnodes")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    if not os.path.exists(f"./storage/vectorindex/crag-subnodes"):
        print('Creating subnodes vector index...\n')
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(all_nodes, storage_context=storage_context)
        vector_index.storage_context.persist(
            persist_dir=f"./storage/vectorindex/crag-subnodes")
    else:
        print('Loading subnodes vector index...\n')
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/vectorindex/crag-subnodes",
            vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)

    return vector_index, all_nodes

# 构造子Node与索引
sub_index, sub_nodes = create_subnodes_index(base_nodes)
# print("----------------------------------------------------------+++")
# print(sub_index)
# print("----------------------------------------------------------+++")

#准备子Node层的检索器
sub_retriever = sub_index.as_retriever(similarity_top_k=2)

#准备一个所有Node的id与Node的对应关系字典
#这个字典用于在递归检索时，根据index_id快速地找到对应的对象
sub_nodes_dict = {n.node_id: n for n in sub_nodes}

####################################子节点索引搜索#######################################
#构造递归检索器
# recursive_retriever = RecursiveRetriever(
#     "root_retriever",
#     retriever_dict={"root_retriever": sub_retriever},
#     node_dict=sub_nodes_dict,
#     verbose=True,
# )

#用递归检索器构造查询引擎
# recursive_query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
#
# #测试
# response = recursive_query_engine.query("请解释下的caffe框架的概念?")
# pprint_response(response)
# print("++++++++++++++++++++++++++++++++++++++++++++++++")
# print(sub_nodes_dict['node_16'].text)
####################################子节点索引搜索#######################################


# 定义中文提示模板
summary_prompt_template = """
    请为以下文本生成一个简洁的摘要：
    {text}

    摘要要求：
    1. 摘要长度不超过 50 字。
    2. 突出文本的核心内容。
    3. 使用简洁明了的语言。

    摘要：
    """

# 根据基础Node构造摘要Node
def create_summary_nodes(base_nodes):
    # 构造一个元数据抽取器
    extractor = SummaryExtractor(summaries=["self"], prompt_template=summary_prompt_template, show_progress=True)
    summary_dict = {}

    # 为了避免重复抽取，进行持久化存储
    if not os.path.exists(f"./storage/metadata/summarys.json"):
        print('Extract new summary...\n')
        # 创建 TextNode
        text_node = TextNode(
            text="大型语言模型（LLM）是人工智能领域的重要研究方向之一。它们通过大量的文本数据进行训练，能够生成高质量的文本内容，广泛应用于机器翻译、文本生成、问答系统等场景。"
        )
        # 抽取元数据，建立从Node到元数据的词典
        summarys = extractor.extract(text_node)
        print(f"summarys: {summarys}")
        for node, summary in zip(base_nodes, summarys):
            summary_dict[node.node_id] = summary

        with open('./storage/metadata/summarys.json', "w") as fp:
            json.dump(summary_dict, fp, ensure_ascii=False)
    else:
        print('Loading summary from storage...\n')
        with open('./storage/metadata/summarys.json', "r") as fp:
            summary_dict = json.load(fp, ensure_ascii=False)

    # 根据摘要构造摘要Node，注意使用IndexNode类型
    all_nodes = []
    for node_id, summary in summary_dict.items():
        all_nodes.append(IndexNode(text=summary["section_summary"], index_id=f"summary_{node_id}"))

        # 加入基础Node
        all_nodes.extend(IndexNode.from_text_node(base_node, f"base_{base_node.node_id}") for base_node in base_nodes)

    # 构造摘要Node层的索引
    collection = chroma.get_or_create_collection(name=f"crag-summarynodes")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if not os.path.exists(f"./storage/vectorindex/crag-summarynodes"):
        print('Creating summary nodes vector index...\n')
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(all_nodes, storage_context=storage_context)
        vector_index.storage_context.persist(persist_dir=f"./storage/vectorindex/crag-summarynodes")
    else:
        print('Loading summary nodes vector index...\n')
        storage_context = StorageContext.from_defaults(persist_dir=f"./storage/vectorindex/crag-summarynodes",
                                                       vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)

    return vector_index, all_nodes

summary_index,summary_nodes = create_summary_nodes(base_nodes)
summary_retriever = summary_index.as_retriever(similarity_top_k=2)
summary_nodes_dict = {n.node_id: n for n in summary_nodes}

#构造一个递归检索器
recursive_retriever = RecursiveRetriever(
    "root_retriever",
    retriever_dict={"root_retriever": summary_retriever},
    node_dict=summary_nodes_dict,
    verbose=True,
)
recursive_query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
response = recursive_query_engine.query("请解释下的caffe框架的概念?")
pprint_response(response)
