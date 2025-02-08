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
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline.components.input import InputComponent

embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

#加载文档，构造向量存储索引
docs = SimpleDirectoryReader(input_files=["./data/xiaomi.txt"]).load_data()
index = VectorStoreIndex.from_documents(docs)

#准备组件
input = InputComponent()
prompt_tmpl = PromptTemplate("对问题进行完善，输出新的问题：{query_str}" )
retriever = index.as_retriever(similarity_top_k=3)
summarizer = get_response_synthesizer(response_mode="tree_summarize")

#构造一个查询管道
p = QueryPipeline(verbose=True)

#把上面构造的模块添加进来
p.add_modules(
    {
        "input": input,
        "prompt": prompt_tmpl,
        "llm":llm,
        "retriever": retriever,
        "summarizer": summarizer,
    }
)

#连接这些模块
p.add_link("input", "prompt")
p.add_link('prompt',"llm")
p.add_link('llm','retriever')
p.add_link("retriever", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")

output = p.run(input='小米手机的优势是什么')
print(output)

