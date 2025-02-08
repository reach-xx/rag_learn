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


embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm


docs = SimpleDirectoryReader(input_files=["./data/yiyan.txt"]).load_data()
nodes = SentenceSplitter(chunk_size=100,chunk_overlap=0).get_nodes_from_documents(docs)
vector_index = VectorStoreIndex(nodes)

retriever =vector_index.as_retriever(similarity_top_k=5)

#直接检索出结果
nodes = retriever.retrieve("百度文心一言的逻辑推理能力怎么样？")
print('================before rerank================')
print(nodes)

#使用Cohere Rerank模型重排序结果
cohere_rerank = CohereRerank(model='rerank-multilingual-v3.0', api_key='706xpu56HqR91XUSPseqwQAufc7dygIYqLIW3v8W', top_n=2)
rerank_nodes = cohere_rerank.postprocess_nodes(nodes, query_str='百度文心一言的逻辑推理能力怎么样？')
print('================after rerank================')
print(rerank_nodes)