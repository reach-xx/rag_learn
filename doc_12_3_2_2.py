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
from typing import Dict
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
from llama_index.core.agent import ReActAgent
from llama_index.core.objects.tool_node_mapping import SimpleToolNodeMapping
from llama_index.core.objects import ObjectIndex
from llama_index.core.llms import ChatMessage, MessageRole
from fastapi.responses import JSONResponse
from fastapi import APIRouter, FastAPI, HTTPException,status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,cast
from multiprocessing.managers import BaseManager
from fastapi.responses import StreamingResponse
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from queue import Queue
from threading import Thread
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
import uvicorn
import os


#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = \
OllamaEmbedding(model_name="nomic-embed-text")

#向量库
chroma = chromadb.HttpClient(host="localhost", port=8080)
chroma.delete_collection(name="doc12")
collection = chroma.get_or_create_collection(name="doc12", metadata={"hnsw:space": "cosine"})

#目录
HOME_DIR='./'
DATA_DIR = f'{HOME_DIR}/data'
STOR_DIR = f'{HOME_DIR}/storage'

#本应用的知识文档
city_docs = {
    "Beijing":f'{DATA_DIR}/beijing.txt',
    "Guangzhou":f'{DATA_DIR}/guangzhou.txt',
    "Nanjing":f'{DATA_DIR}/nanjing.txt',
    "Shanghai":f'{DATA_DIR}/shanghai.txt'
}

#创建针对单个文档的Tool Agent
def _create_doc_agent(name:str,callback_manager: CallbackManager):

    file = city_docs[name]
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

    query_engine = vector_index.as_query_engine(similarity_top_k=5)

    # 构造摘要索引与查询引擎
    summary_index = SummaryIndex(nodes)
    summary_engine = summary_index.as_query_engine(
                            response_mode="tree_summarize")

    # 把两个查询引擎工具化
    query_tool = QueryEngineTool.from_defaults(query_engine=query_engine,
                                               name=f'query_tool',
                                               description=f'用于回答关于城市{name}的具体问题，包括经济、旅游、文化、历史等方面')
    summary_tool = QueryEngineTool.from_defaults(query_engine=summary_engine,
                                                 name=f'summary_tool',
                                                 description=f'任何需要对城市{name}的各个方面进行全面总结的请求请使用本工具。如果您想查询有关 {name} 的某些详细信息，请使用query_tool')

    city_tools = [query_tool,summary_tool]

    # 使用两个工具创建单独的文档Agent
    doc_agent = ReActAgent.from_tools(city_tools,
                                      verbose=True,
                                      system_prompt=f'你是一个专门设计用于回答有关城市{name}信息查询的助手。在回答问题时，你必须始终使用至少一个提供的工具；不要依赖先验知识。不要编造答案。',
                                      callback_manager=callback_manager)
    return doc_agent

#循环创建针对所有文档的Tool Agent，并将其保存到doc_agents_dict字典中
def _create_doc_agents(callback_manager: CallbackManager):

    print('Creating document agents for all citys...\n')
    doc_agents_dict = {}
    for city in city_docs.keys():
        doc_agents_dict[city] = _create_doc_agent(city,callback_manager)

    return doc_agents_dict

def _create_top_agent(doc_agents: Dict,callback_manager: CallbackManager):
    all_tools = []
    for city in doc_agents.keys():
        city_summary = (
            f" 这部分包含了有关{city}的城市信息. "
            f" 如果需要回答有关{city}的任务问题，请使用这个工具.\n"
        )

        #把创建好的每个Tool Agent都工具化
        doc_tool = QueryEngineTool(
            query_engine=doc_agents[city],
            metadata=ToolMetadata(
                name=f"tool_{city}",
                description=city_summary,
            ),
        )
        all_tools.append(doc_tool)

    #实现一个对象索引，用于检索工具
    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    if not os.path.exists(f"{STOR_DIR}/top"):
        storage_context = StorageContext.from_defaults()
        obj_index = ObjectIndex.from_objects(
            all_tools,
            tool_mapping,
            VectorStoreIndex,
            storage_context=storage_context
        )
        storage_context.persist(persist_dir=f"{STOR_DIR}/top")
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{STOR_DIR}/top"
        )
        index = load_index_from_storage(storage_context)
        obj_index = ObjectIndex(index, tool_mapping)

    print('Creating top agent...\n')

    #创建Top Agent
    top_agent = ReActAgent.from_tools(
                          tool_retriever=obj_index.as_retriever(similarity_top_k=3),
                          verbose=True,
                          system_prompt="你是一个被设计来回答关于一组给定城市查询的助手。请始终使用提供的工具来回答一个问题。不要依赖先验知识。不要编造答案",
                          callback_manager=callback_manager)
    return top_agent

def get_agent():

    # 构造并加入自定义的事件处理器
    queue = Queue()
    #创建Agent，此处暂时忽略callback_manager
    handler = StreamingCallbackHandler(queue)
    callback_manager = CallbackManager([handler])

    doc_agents = _create_doc_agents(callback_manager)
    top_agent = _create_top_agent(doc_agents,callback_manager)

    return top_agent

#单个消息：角色与内容
class _Message(BaseModel):
    role: MessageRole
    content: str

#接口数据：消息的顺序列表
class _ChatData(BaseModel):
    messages: List[_Message]


chat_router = r = APIRouter()


@r.post("/nostream")
async def chat_nostream(
        data: _ChatData
):
    # 获得Agent
    agent = get_agent()

    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )

    # 最后一个产生消息的角色必须是user
    lastMessage = data.messages.pop()
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )

    # 创建消息历史
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]

    # 调用Agent获得响应结果并返回
    chat_result = agent.chat(lastMessage.content, messages)
    return JSONResponse(content={"text": str(chat_result)}, status_code=200)

def decode_sse_messages(content):
    # 解码 SSE 消息的逻辑
    messages = []
    for line in content.split('\n'):
        if line.startswith('data:'):
            messages.append(line[len('data:'):].strip())
    return messages

def convert_sse(obj):
    return "data: {}\n\n".format(json.dumps(obj, ensure_ascii=False))

@r.post("")
async def chat(
        data: _ChatData
):
    # 获得Agent
    agent = get_agent()

    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )

    # 最后一个产生消息的角色必须是user
    lastMessage = data.messages.pop()
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )

    # 创建消息历史
    messages = [
        ChatMessage(
            role=m.role,
            content=decode_sse_messages(m.content),
        )
        for m in data.messages
    ]

    # # 调用Agent获得响应结果并返回
    # chat_result = agent.stream_chat(lastMessage.content, messages)
    # # print(f"chat_result: {chat_result.response_gen}")
    # # for token in chat_result.response_gen:
    # #     print(token)
    # # 构造一个生成器，用于迭代处理输出的token
    # def event_generator():
    #     for token in chat_result.response_gen:
    #         yield convert_sse(token)
    thread = Thread(target=agent.stream_chat,
                    args=(lastMessage.content, messages))
    thread.start()

    # 生成器，用于构造StreamingResponse对象
    def event_generator():

        # 从队列中读取对象
        queue = agent.callback_manager.handlers[0].queue

        while True:
            next_item = queue.get(True, 60.0)

            # 判断next_item
            # 如果是EventObject类型的，则是FUNCTION_CALL事件
            if isinstance(next_item, EventObject):
                yield convert_sse(dict(next_item))

            # 如果是StreamingAgentChatResponse类型的，则是AGENT_STEP事件
            elif isinstance(next_item, StreamingAgentChatResponse):
                response = cast(StreamingAgentChatResponse, next_item)

                # 通过response_gen方法迭代处理流式响应
                for token in response.response_gen:
                    yield convert_sse(token)
                break

    # 客户端流式响应
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# if __name__ == '__main__':
#
#     top_agent = get_agent()
#     print('Starting to stream chat...\n')
#     streaming_response = top_agent.streaming_chat_repl()

#自定义一个事件类型，包含类型与事件信息
class EventObject(BaseModel):
    type: str
    payload: dict

#自定义流跟踪的事件处理器
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue) -> None:
        super().__init__([], [])
        self._queue = queue

    #自定义on_event_start接口
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:

        #跟踪Agent的工具使用
        if event_type == CBEventType.FUNCTION_CALL:
            self._queue.put(
                EventObject(
                    type="function_call",
                    payload={
                        "arguments_str": str(payload["function_call"]),
                        "tool_str": str(payload["tool"].name),
                    },
                )
            )

    #自定义on_event_end接口
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:

        #跟踪Agent的工具调用
        if event_type == CBEventType.FUNCTION_CALL:
            self._queue.put(
                EventObject(
                    type="function_call_response",
                    payload={"response": payload["function_call_response"]},
                )
            )

        #跟踪Agent每个步骤的大模型响应
        elif event_type == CBEventType.AGENT_STEP:
            self._queue.put(payload["response"])

    @property
    def queue(self) -> Queue:
        """Get the queue of events."""
        return self._queue

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

#将chat_router路由包含进来，有利于路由的模块化组织管理
app.include_router(chat_router, prefix="/api/chat")

if __name__ == "__main__":
    uvicorn.run(app="doc_12_3_2_2:app", host="0.0.0.0", port=8090, reload=True)
    # message = "hello world! I love you!"
    # print(message)
    # encode = convert_sse(message)
    # print(f"encode: {encode}")
    # decode = decode_sse_messages(encode)
    # print(f"decode:  {decode}")
