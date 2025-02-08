from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core import Document,SimpleDirectoryReader,Settings,load_index_from_storage,get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core.tools import ToolMetadata
import pprint
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index.core import PropertyGraphIndex,TreeIndex,KeywordTableIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import VectorStoreIndex,SummaryIndex,SimpleKeywordTableIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.response.pprint_utils import pprint_response

embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

docs = SimpleDirectoryReader(input_files=["./data/xiaomi.txt"]).load_data()

summary_index =\
SummaryIndex.from_documents(docs,chunk_size=100,chunk_overlap=0)
vector_index =\
VectorStoreIndex.from_documents(docs,chunk_size=100,chunk_overlap=0)
keyword_index =\
SimpleKeywordTableIndex.from_documents(docs,chunk_size=100,chunk_overlap=0)

#构造3个可用工具
summary_tool = QueryEngineTool.from_defaults(
query_engine=summary_index.as_query_engine(response_mode="tree_summarize",),
    description=(
        "有助于总结与小米手机相关的问题"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(),
    description=(
        "适合检索与小米手机相关的特定上下文"
    ),
)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_index.as_query_engine(),
    description=(
        "适合使用关键词从文章中检索特定的上下文"
    ),
)

#构造可多选的路由查询引擎
query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,vector_tool,keyword_tool
    ],verbose=True
)
response = query_engine.query("小米手机的处理器是什么？")
pprint_response(response, show_source=True)