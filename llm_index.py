import chromadb

from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core import SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType
)
from pprint import pprint
from llama_index.core import PromptTemplate

#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = \
OllamaEmbedding(model_name="herald/dmeta-embedding-zh")

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager


#加载与读取文档
reader = SimpleDirectoryReader(input_files=["./data/zengguofan.txt","./data/UNIX_introduction.txt"])
documents = reader.load_data()

#分割文档
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False, callback_manager=callback_manager)

#准备向量存储
chroma = chromadb.HttpClient(host="localhost", port=8080)
chroma.delete_collection(name="ragdb")
collection = chroma.get_or_create_collection(name="ragdb", metadata={"hnsw:space": "cosine"})
vector_store = ChromaVectorStore(chroma_collection=collection)

#准备向量存储索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)



#构造查询引擎
query_engine = index.as_query_engine()
prompts_dict = query_engine.get_prompts()
pprint(prompts_dict.keys())

pprint(prompts_dict["response_synthesizer:text_qa_template"].get_template())
my_qa_prompt_tmpl_str = (
    "以下是上下文信息。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据上下文信息回答问题，不要依赖预置知识，不要编造。\n"
    "问题: {query_str}\n"
    "回答: "
)
my_qa_prompt_tmpl = PromptTemplate(my_qa_prompt_tmpl_str)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": my_qa_prompt_tmpl}
)

prompts_dict = query_engine.get_prompts()
pprint(prompts_dict["response_synthesizer:text_qa_template"].get_template())

while True:
    user_input = input("问题：")
    if user_input.lower() == "exit":
        break

    response = query_engine.query(user_input)
    pprint(llama_debug.get_event_pairs(CBEventType.QUERY))
    print("AI助手：", response.response)

