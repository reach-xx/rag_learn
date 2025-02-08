from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from llama_index.core.schema import IndexNode, TextNode
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding



#设置模型
Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = \
OllamaEmbedding(model_name="nomic-embed-text")

# 定义提示模板
summary_prompt_template = """
请为以下文本生成一个简洁的摘要：
{text}

摘要要求：
1. 摘要长度不超过 50 字。
2. 突出文本的核心内容。
3. 使用简洁明了的语言。

摘要：
"""

# 创建 SummaryExtractor
extractor = SummaryExtractor(
    summaries=["self"],  # 提取自身摘要
    prompt_template=summary_prompt_template,  # 使用自定义提示模板
    show_progress=True  # 显示进度
)

# 创建 TextNode
text_node = TextNode(
    text="大型语言模型（LLM）是人工智能领域的重要研究方向之一。它们通过大量的文本数据进行训练，能够生成高质量的文本内容，广泛应用于机器翻译、文本生成、问答系统等场景。"
)

# 提取摘要
summary = extractor.extract([text_node])  # 注意：extract 接受一个列表

# 打印摘要
print("生成的摘要：")
print(summary)
print(summary_prompt_template)
