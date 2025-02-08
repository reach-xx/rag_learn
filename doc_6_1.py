from llama_index.core import Document,SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import pprint
from llama_index.core.schema import MetadataMode

llm = Ollama(model='qwen2.5')
embedded_model = \
OllamaEmbedding(model_name="herald/dmeta-embedding-zh",
embed_batch_size=50)
docs = SimpleDirectoryReader(input_files=["./data/unix_standard.txt"]).load_data()

#构造一个数据摄取管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
        TitleExtractor(llm=llm, show_progress=False)
    ]
)

#运行这个数据摄取管道
nodes = pipeline.run(documents=docs)
embeddings = embedded_model.get_text_embedding_batch([node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],show_progress=True)

for node, embedding in zip(nodes, embeddings):
    node.embedding = embedding

print("emedding: ", len(nodes[0].embedding))

print(nodes)