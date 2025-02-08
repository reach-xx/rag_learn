from llama_index.core import Document,SimpleDirectoryReader,Settings,load_index_from_storage,get_response_synthesizer,PromptTemplate
from llama_index.core.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import pprint
import os
from llama_index.core.schema import MetadataMode
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core import PropertyGraphIndex,TreeIndex,KeywordTableIndex
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from typing import List
from llama_index.core.bridge.pydantic import BaseModel, Field



#此处使用内置的LlamaDebugHandler处理器进行跟踪（也可以使用Langfuse平台跟踪）
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm


#这里给Prompt模板增加一个language_name参数
qa_prompt_tmpl = (
    "根据以下上下文信息：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "使用{language_name}回答以下问题\n "
    "问题: {query_str}\n"
    "答案: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)


class Phone(BaseModel):
    name: str
    description: str
    features: List[str]

#构造refine响应生成器
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    summary_template=qa_prompt)

#模拟检索出的3个Node
nodes = [NodeWithScore(node=Node(text="小麦手机是一款专为满足现代生活需求而设计的智能手机。它的设计简洁大方，线条流畅，给人一种优雅的感觉"), score=1.0),
         NodeWithScore(node=Node(text="小麦手机采用了最新的处理器技术，运行速度快，性能稳定，无论是玩游戏、看电影还是处理工作，都能轻松应对"), score=1.0),
         NodeWithScore(node=Node(text="小麦手机还配备了高清大屏，色彩鲜艳，画面清晰，无论是阅读、浏览网页还是观看视频，都能带来极佳的视觉体验"), score=1.0)
         ]

#把问题和Node交给响应生成器响应生成
response = response_synthesizer.synthesize(
    "介绍一下小麦手机的优点",
    nodes=nodes,
    language_name = "中文"
)

print(response)

def print_events_llm():
    events = llama_debug.get_event_pairs('llm')

    #发生了多少次大模型调用
    print(f'Number of LLM calls: {len(events)}')

    #依次打印所有大模型调用的消息
    for i,event in enumerate(events):
        print(f'\n=========LLM call {i+1} messages===========')
        pprint.pprint(event[1].payload["messages"])
        print(f'\n=========LLM call {i+1} response============')
        pprint.pprint(event[1].payload["response"].message.content)

print_events_llm()