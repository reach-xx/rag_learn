from llama_index.core.schema import Document,MetadataMode
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.llms.ollama import Ollama
import pprint

doc = Document(text='RAG是一种常见的大模型应用范式，它通过检索—排序—生成的方式生成文本。',metadata={'title':'RAG模型介绍','author':'llama-index'})
# pprint.pprint(doc.dict())


doc4 = Document(text = "百度是一家中国的搜索引擎公司。",
                metadata={
                    "file_name": "test.txt",
                    "category": "technology",
                    "author": "random person",
                    },
                excluded_llm_metadata_keys=["file_name"],
                excluded_embed_metadata_keys=["file_name", 'author'],
                metadata_seperator=" | ",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
                )

# print("\n全部元数据: \n",
#       doc4.get_content(metadata_mode=MetadataMode.ALL))
#
# print("\n嵌入模型看到的: \n",
#       doc4.get_content(metadata_mode=MetadataMode.EMBED))
#
# print("\n大模型看到的: \n",
#       doc4.get_content(metadata_mode=MetadataMode.LLM))
#
# print("\n没有元数据: \n",
#       doc4.get_content(metadata_mode=MetadataMode.NONE))

#================数据连接器加载数据生成的Document对象===================
docs2 = SimpleDirectoryReader(input_files=["./data/unix_standard.txt"]).load_data()
# print("number of documents in docs2 is : ", len(docs2))
# print("\n\ncontent: \n\n",docs2[0].text)
# print(f"metadata: {docs2[0].metadata}")


#================使用Document对象生成Node对象==========================
docs = [Document(text='AIGC是一种利用人工智能技术自动生成内容的方法，这些内容可以包括文本、音频、图像、视频、代码等多种形式。\n AIGC的发展得益于深度学习技术的进步，特别是自然语言处理领域的成就，使得计算机能够更好地理解语言并实现自动化内容生成。')]
#构造一个简单的数据分割器
parser = TokenTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
nodes = parser.get_nodes_from_documents(docs)

# print(f"docs: {docs[0].metadata}")
#
# for i, node in enumerate(nodes):
#     print(f"Node {i}: {node}")
#     print(f"Node {i}: {node.text}")
#
#
# print(f"Node1: {nodes[0].relationships}")
# print(f"Node2: {nodes[1].relationships}")


#===================摘要抽取器 SummaryExtractor=======================
llm = Ollama(model="qwen2.5")

docs = SimpleDirectoryReader(input_files=["./data/unix_standard.txt"]).load_data()
summary_extractor = SummaryExtractor(llm=llm, show_progress=False, prompt_template="请生成以下内容的中文摘要：{context_str} \n 摘要:,", metadata_mode = MetadataMode.NONE)
print(summary_extractor.extract(docs))


#===================问答抽取器 QuestionsAnsweredExtractor================

template = '''
以下是上下文：
{context_str}

给定上下文信息\
生成此上下文可以提供的{num_questions}问题\
在其他地方不太可能找到具体的答案。

可以提供更高级别的周围环境摘要\
也。尝试使用这些摘要来生成更好的问题\
这个上下文可以回答。
'''

questions_extractor = \
QuestionsAnsweredExtractor(llm=llm, prompt_template=template, show_progress=False, metadata_mode=MetadataMode.NONE)
print(questions_extractor.extract(docs))













