# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from llama_index.core import Settings

#ollama嵌入模型
# embed_model = OllamaEmbedding("herald/dmeta-embedding-zh")

#TEI的本地嵌入模型加入 huggingface上的模型
embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-large-zh-v1.5",
    timeout=60,  # timeout in seconds
    embed_batch_size=10,  # batch size for embedding
)
print(embed_model)

# embeddings1 = embed_model.get_text_embedding(
#     "中国的首都是北京"
# )
# embedding2 = embed_model.get_text_embedding(
#     "中国的首都是哪里？"
# )
# embedding3 = embed_model.get_text_embedding(
#     "苹果是一种好吃的水果"
# )
# print(embed_model.similarity(embeddings1, embedding2))
# print(embed_model.similarity(embeddings1, embedding3))