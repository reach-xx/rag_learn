from llama_index.core.node_parser import SentenceSplitter,TokenTextSplitter,SentenceWindowNodeParser
from llama_index.core.schema import Document,MetadataMode
from llama_index.core.node_parser.text.utils import (
    split_by_char,
    split_by_regex,
    split_by_sentence_tokenizer,
    split_by_sep,
)

splitter = SentenceSplitter(separator="\n")
nodes = splitter.get_nodes_from_documents([Document(text="This is a test.\n haha!")])
# print(nodes)

fn = split_by_sep("\n")
result = fn('Google公司介绍 \n Google是一家搜索引擎与云计算公司，总部位于美国加利福尼亚州山景城。主要产品是搜索引擎、广告服务、企业服务、云计算等。')
# print(result)
print("Size of the result array:", len(result))

fn = split_by_sentence_tokenizer()
result = fn('Google公司介绍 \n Google是一家搜索引擎与云计算公司，总部位于美国加利福尼亚州山景城。主要产品是搜索引擎、广告服务、企业服务、云计算等。')
# print(result)
print("Size of the result array:", len(result))

fn = split_by_regex("[^，,.;。？！]+[，,.;。？！]?")
result = fn('Google公司介绍 \n Google是一家搜索引擎与云计算公司，总部位于美国加利福尼亚州山景城。主要产品是搜索引擎、广告服务、企业服务、云计算等。')
# print(result)
print("Size of the result array:", len(result))

fn = split_by_char()
result = fn('Google公司介绍 \n Google是一家搜索引擎与云计算公司，总部位于美国加利福尼亚州山景城。主要产品是搜索引擎、广告服务、企业服务、云计算等。')
# print(result)
print("Size of the result array:", len(result))

docs = [Document(text="Google公司介绍 \n Google是一家搜索引擎与云计算公司 \n 总部位于美国加利福尼亚州山景城。主要产品是搜索引擎、广告服务、企业服务、云计算等。")]
splitter = TokenTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separator="\n",
    backup_separators=["。"]
)
nodes = splitter.get_nodes_from_documents(docs )
# print(nodes)


import re
#自定义一个分割文本的函数
def my_chunking_tokenizer_fn(text:str):
    #跟踪是否进入本方法
    print('start my chunk tokenizer function...')
    sentence_delimiters = re.compile(u'[。！？]')
    sentences = sentence_delimiters.split(text)
    return [s.strip() for s in sentences if s]

docs = [Document(text="***Google公司介绍***Google是一家搜索引擎与云计算公司。总部位于美国加利福尼亚州山景城。主要产品是搜索引擎、广告服务、企业服务、云计算等。")]
node_parser = SentenceSplitter(chunk_size=50,
                             chunk_overlap=0,
                             paragraph_separator="***",
                              chunking_tokenizer_fn=my_chunking_tokenizer_fn,
                             secondary_chunking_regex = "[^,.;。？！]+[,.;。？！]?",
                             separator="\n")
nodes = node_parser.get_nodes_from_documents(docs)
# print(nodes)

print("------------------------------------------------")
docs = [Document(text="Google公司介绍:Google是一家搜索引擎与云计算公司。\
                      总部位于美国加利福尼亚州山景城。\
                      主要产品是搜索引擎、广告服务、企业服务、云计算等。\
                      百度是一家中国的搜索引擎公司。")]
splitter = SentenceWindowNodeParser(
    window_size=2,
    sentence_splitter = my_chunking_tokenizer_fn
)
nodes = splitter.get_nodes_from_documents(docs)
# print(f"nodes num: {len(nodes)}")
# print(nodes)



#=-=======================HierarchicalNodeParser===============
from llama_index.core.node_parser import HierarchicalNodeParser

docs = [Document(text="Google公司介绍:Google是一家搜索引擎与云计算公司。\
                      总部位于美国加利福尼亚州山景城。\
                      Google公司成立于1998年9月4日，由拉里·佩奇和谢尔盖·布林共同创立。\
                      主要产品是搜索引擎、广告服务、企业服务、云计算等。\
                      百度是一家中国的搜索引擎公司。\
                      百度公司成立于2000年1月1日，由李彦宏创立。")]
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 100, 50]
)
nodes = node_parser.get_nodes_from_documents(docs)
print("nums: ", len(nodes))
# print(nodes)


#=========================SemanticSplitterNodeParser======================================
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding

docs = SimpleDirectoryReader(input_files=["./data/unix_standard.txt"]).load_data()
embed_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh")
splitter = SemanticSplitterNodeParser(
    breakpoint_percentile_threshold=20,
    sentence_splitter = my_chunking_tokenizer_fn,
    embed_model=embed_model
)
nodes = splitter.get_nodes_from_documents(docs)
# print(len(nodes))
# print(nodes)

#==============================分割特殊的文本=============================
from llama_index.readers.file.flat import FlatReader
from llama_index.core.node_parser import CodeSplitter
from pathlib import Path
# #分割Markdown格式的文档
# docs = FlatReader().load_data(Path("../../data/5-python.md"))
# markdown_parser = MarkdownNodeParser()
# nodes = markdown_parser.get_nodes_from_documents(docs )
# print(nodes)
#
# #分割HTML格式的文档
# docs = FlatReader().load_data(Path("../../data/10-google.html"))
# html_parser = HTMLNodeParser()
# nodes = html_parser.get_nodes_from_documents(docs )
# print(nodes)
#
# #分割JSON格式的文档
# docs = FlatReader().load_data(Path("../../data/11-quantum.json"))
# json_parser = JSONNodeParser()
# nodes = json_parser.get_nodes_from_documents(docs )
# print(nodes)

#分割源代码文档
docs = FlatReader().load_data(Path("splitter.py"))
code_parser = CodeSplitter(language="python")
nodes = code_parser.get_nodes_from_documents(docs)
print(nodes)



