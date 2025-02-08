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
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler

langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key="pk-lf-78690601-cc01-446e-bebb-2410bf5b737b",
    secret_key="sk-lf-307a97cd-54c6-4820-9000-6c8bf84f838c",
    host="https://cloud.langfuse.com"
)

Settings.callback_manager = CallbackManager([langfuse_callback_handler])

embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

#构造SQL查询引擎
engine =\
create_engine("postgresql://postgres:123456@localhost:5432/postgres")

#构造SQLDatabase对象
sql_database = SQLDatabase(engine, include_tables=["orders"])

query_engine = NLSQLTableQueryEngine(
                sql_database=sql_database,
                tables=["orders"],
                llm=llm
)

#测试
response = query_engine.query("所有产品一共销售了多少钱")
print(response)
langfuse_callback_handler.flush()