from llama_index.core import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Document,SimpleDirectoryReader,Settings

#model
embedded_model = OllamaEmbedding(model_name="herald/dmeta-embedding-zh", embed_batch_size=50)
llm = Ollama(model='qwen2.5')

Settings.embed_model=embedded_model
Settings.llm = llm


template = (
    "以下是提供的上下文信息：\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "根据这些信息，请回答以下问题：{query_str}\n"
)
qa_template = PromptTemplate(template)

prompt = qa_template.format(context_str='小麦15 PRO是小麦公司最新推出的6.7寸大屏旗舰手机。', query_str='小麦15pro的屏幕尺寸是多少？')
print(prompt)

messages = qa_template.format_messages(context_str='小麦15 PRO是小麦公司最新推出的6.7寸大屏旗舰手机。', query_str='小麦15pro的屏幕尺寸是多少？')
print(messages)

