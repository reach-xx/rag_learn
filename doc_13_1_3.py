from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core import VectorStoreIndex,SummaryIndex,SimpleKeywordTableIndex
from typing import Any, Callable, Optional, Sequence,List
from llama_index.core import Document,SimpleDirectoryReader
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.response.pprint_utils import pprint_response


#设置模型
llm = Ollama(model="qwen2.5")
Settings.llm = llm
Settings.embed_model = \
OllamaEmbedding(model_name="nomic-embed-text")

# 生成答案
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "以下是上下文\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请仅根据上面的上下文，回答以下问题，不要编造其他内容。\n"
    "如果上下文中不存在相关信息，请拒绝回答。\n"
    "问题: {query_str}\n"
    "答案: "
)
text_qa_prompt = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)

# 评估相关性
EVALUATE_PROMPT_TEMPLATE = """您是一个评分人员，评估检索出的文档与用户问题的相关性。
        以下是检索出的文档：
        ----------------
        {context}
        ----------------

        以下是用户问题：
        ----------------
        {query_str}
        ----------------
        如果文档中包含与用户问题相关的关键词或语义，且有助于解答用户问题，请将其评为相关。
        请给出yes或no来表明文档是否与问题相关。
        注意只需要输出yes或no，不要有多余解释。
        """
evaluate_prompt = PromptTemplate(EVALUATE_PROMPT_TEMPLATE)

# 重写输入问题
REWRITE_PROMPT_TEMPLATE = """你需要生成对检索进行优化的问题。请根据输入内容，尝试推理其中的语义意图/含义。
        这是初始问题：
        ----------------
        {query_str}
        ----------------
        请提出一个改进的问题："""
rewrite_prompt = PromptTemplate(REWRITE_PROMPT_TEMPLATE)


# 构造检索器
def create_retriever(file):
    docs = SimpleDirectoryReader(input_files=[file]).load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index.as_retriever(similarity_top_k=3), index.as_query_engine()


# 评估检索结果
def evaluate_nodes(query_str: str, retrieved_nodes: List[Document]):
    # 构造一个用于评估的简单查询管道，直接使用大模型也一样
    evaluate_pipeline = QueryPipeline(chain=[evaluate_prompt, llm])

    filtered_nodes = []
    need_search = False
    for node in retrieved_nodes:

        # 对Node中的内容与输入问题评估相关性
        relevancy = evaluate_pipeline.run(
            context=node.text, query_str=query_str
        )

        # 如果相关，则返回；否则，需要搜索
        if (relevancy.message.content.lower() == 'yes'):
            filtered_nodes.append(node)
        else:
            need_search = True

    return filtered_nodes, need_search

#重写输入问题
def rewrite(query_str: str):
    new_query_str = llm.predict(
        rewrite_prompt, query_str=query_str
    )
    return new_query_str

#搜索
def web_search(query_str:str):
    tavily_tool = TavilyToolSpec(api_key="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")
    search_results = tavily_tool.search(query_str,max_results=5)
    return "\n".join([result.text for result in search_results])

#使用大模型直接生成答案
def query(query_str,context_str):
    response = llm.predict(
        text_qa_prompt, context_str=context_str, query_str=query_str
    )
    return response

file_name = "./data/nanjing.txt"

#构造检索器与查询引擎
retriever, query_engine = create_retriever(file_name)
query_str = '北京市的人口数量是多少与分布情况如何？参加2024年中考的学生数量是多少？'

#先测试直接生成答案
response = query_engine.query(query_str)
print(f'-----------------Response from query engine-------------------')
pprint_response(response, show_source=True)

#测试C-RAG流程
#C-RAG：检索
print(f'-----------------Response from CRAG-------------------')
retrieved_nodes = retriever.retrieve(query_str)
print(f'{len(retrieved_nodes)} nodes retrieved.\n')

#C-RAG：评估检索结果，仅保留相关的上下文
filtered_nodes,need_search = evaluate_nodes(query_str,retrieved_nodes)
print(f'{len(filtered_nodes)}   need_search:{need_search} nodes relevant.\n')
filtered_texts = [node.text for node in filtered_nodes]
filtered_text = "\n".join(filtered_texts)
print(f"filtered_text: {filtered_text}")

#C-RAG：如果存在不相关知识，那么重写输入问题并借助网络搜索
if need_search:
    new_query_str = rewrite(query_str)
    search_text = web_search(new_query_str)
    print(f"search_text: {search_text}")

#组合成新的上下文，并进行生成
context_str = filtered_text + "\n" + search_text
print(f"context_str: {context_str}")
response = query(query_str, context_str)
print(f'Final Response from crag: \n{response}')




