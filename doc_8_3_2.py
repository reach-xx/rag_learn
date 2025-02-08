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
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector

embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

docs_yiyan = SimpleDirectoryReader(input_files=["./data/yiyan.txt"]).load_data()
docs_nanjing = SimpleDirectoryReader(input_files=["./data/nanjing.txt"]).load_data()

vectorindex_nanjing = VectorStoreIndex.from_documents(docs_nanjing)
query_engine_nanjing = vectorindex_nanjing.as_query_engine()

vectorindex_yiyan = VectorStoreIndex.from_documents(docs_yiyan)
query_engine_yiyan = vectorindex_yiyan.as_query_engine()

#构造第一个工具
tool_nanjing = QueryEngineTool.from_defaults(
    query_engine=query_engine_nanjing,
    description="查询南京市的信息",
)

#构造第二个工具
tool_yiyan = QueryEngineTool.from_defaults(
    query_engine=query_engine_yiyan,
    description="用于查询文心一言的信息",
)

#构造路由模块
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),   #选择器
    query_engine_tools=[				#候选工具
        tool_nanjing, tool_yiyan
    ]
)

#像使用查询引擎一样使用即可
response = query_engine.query("介绍下南京这个城市的情况")
print(response)