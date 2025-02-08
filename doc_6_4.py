from llama_index.core import Document,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import pprint
from llama_index.core.schema import MetadataMode
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.indices.document_summary import DocumentSummaryIndex


#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm

#构造一个向量存储对象用于存储最后输出的Node对象
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(name="pipeline")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)



docs = [Document(text="小麦智能健康手环是一款...",metadata={"title":"智家机器人"},doc_id="doc1"),Document(text="速达飞行者...",metadata={"title":"速达飞行者"},doc_id="doc2")]
doc_summary_index = DocumentSummaryIndex.from_documents(docs, storage_context=storage_context, summary_query = "用中文描述所给文本的主要内容，同时描述这段文本可以回答的一些问题。")
pprint.pprint(doc_summary_index.get_document_summary("doc1"))

