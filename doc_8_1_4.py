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

# llm = OpenAI()
#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm


question_gen_prompt_templ = """
你可以访问多个工具，每个工具都代表一个不同的数据源或API。
每个工具都有一个名称和一个描述字段，格式为JSON字典。
字典的键(key)是工具的名称，值(value)是描述。
你的目的是通过生成一系列可以由这些工具回答的子问题来帮助回答一个复杂的用户问题。

在完成任务时，请考虑以下准则：

	•	尽可能具体
	•	子问题应与用户问题相关
	•	子问题应可通过提供的工具回答
	•	你可以为每个工具都生成多个子问题
	•	工具必须用它们的名称而不是描述来指定
	•	如果你认为不相关，就不需要使用工具

通过调用SubQuestionList函数输出子问题列表。

## Tools
```json
{tools_str}
```

## User Question
{query_str}

"""

#rewriter
question_rewriter = LLMQuestionGenerator.from_defaults(llm=llm)

#转换Prompt
new_prompt = PromptTemplate(question_gen_prompt_templ)
question_rewriter.update_prompts({'question_gen_prompt':new_prompt})

#可用的工具，注意这里只是提供工具的元数据，并未真正提供工具
tool_choices = [
    ToolMetadata(
        name="query_tool_beijing",
        description=(
            "用于查询北京市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
    ToolMetadata(
        name="query_tool_shanghai",
        description=(
            "用于查询上海市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
]

# print('-------------------------')
# query_str = "北京与上海的人口差距是多少？它们的面积相差多少？"
#
# #使用generate方法生成子问题
# choices = question_rewriter.generate(
#                                    tool_choices,
#                                    QueryBundle(query_str=query_str))
#
# pprint.pprint(choices)

def create_city_engine(city:str):
    if city == "南京市":
        documents = SimpleDirectoryReader(
            input_files=["./data/nanjing.txt"],
        ).load_data()
    else:
        documents = SimpleDirectoryReader(
            input_files=["./data/shanghai.txt"],
        ).load_data()
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=0)
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    # print(index.index_struct)
    query_engine = index.as_query_engine()
    return query_engine



#构造两个城市信息查询引擎
query_engine_nanjing = create_city_engine('南京市')
query_engine_shanghai = create_city_engine('上海市')

#查询引擎作为工具
query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine_nanjing,
        metadata=ToolMetadata(
        name="query_tool_nanjing",
        description="用于查询南京市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
    QueryEngineTool(
        query_engine=query_engine_shanghai,
        metadata=ToolMetadata(
        name="query_tool_shanghai",
        description="用于查询上海市各个方面的信息，如基本信息、旅游指南、城市历史等"
        ),
    ),
]

#构造子问题查询引擎
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

print("-------------------------------------------------")
#查询
response = query_engine.query(
    "北京与上海的人口差距是多少？GDP大约相差多少？使用中文回答"
)

print(response)
print("-------------------------------------------------")