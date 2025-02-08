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


#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = \
OllamaEmbedding(model_name="herald/dmeta-embedding-zh")

#向量库
chroma = chromadb.HttpClient(host="localhost", port=8080)
collection = chroma.get_or_create_collection("ragdb")

vector_store = ChromaVectorStore(chroma_collection=collection)

#准备需要的知识文档
eval_documents = \
SimpleDirectoryReader("./data/source_files/").load_data()

#准备用于评估的数据集
eval_questions = \
LabelledRagDataset.from_json("./data/rag_dataset.json")

#准备评估组件
faithfulness = FaithfulnessEvaluator()
relevancy = RelevancyEvaluator()
correctness = CorrectnessEvaluator()
import time
def evaluate_response_time_and_accuracy(chunk_size, num_questions=100):
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0
    total_correctness = 0

    # 构造查询引擎
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
    nodes = \
        node_parser.get_nodes_from_documents(eval_documents, show_progress=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print(f"\nTotal nodes: {len(nodes)}")
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    query_engine = vector_index.as_query_engine(top_K=3)

    # 对指定问题集进行一次评估
    for index, question in enumerate(eval_questions.examples[:num_questions]):
        print(f"\nStart evaluating question【{index}】: {question.query}")
        start_time = time.time()
        response_vector = query_engine.query(question.query)

        # 计算响应时间
        elapsed_time = time.time() - start_time

        # 评估忠实度
        faithfulness_result = faithfulness.evaluate_response(
            query=question.query, response=response_vector
        ).score

        # 评估相关性
        relevancy_result = relevancy.evaluate_response(
            query=question.query, response=response_vector
        ).score

        # 评估正确性，此处需要提供参考答案
        correctness_result = correctness.evaluate_response(
            query=question.query, response=response_vector,
            reference=question.reference_answer
        ).score

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result
        total_correctness += correctness_result

        print(
            f"Response time: {elapsed_time:.2f}s, Faithfulness: {faithfulness_result}, Relevancy: {relevancy_result}, Correctness: {correctness_result}")

    # 计算平均得分
    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions
    average_correctness = total_correctness / num_questions

    return average_response_time, average_faithfulness, average_relevancy, average_correctness


for chunk_size in [128,1024,2048]:
  avg_time, avg_faithfulness, avg_relevancy,average_correctness = \
  evaluate_response_time_and_accuracy(chunk_size)
  print(f"Chunk size {chunk_size} \n Average Response time: {avg_time:.2f}s \n"
        f"Average Faithfulness: {avg_faithfulness:.2f}\n"
        f"    Average Relevancy: {avg_relevancy:.2f}\n"
        f"        Average Correctness: {average_correctness:.2f}")




