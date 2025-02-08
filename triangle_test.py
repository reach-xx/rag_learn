import spacy

# 1. 加载 spaCy 模型
nlp = spacy.load("en_core_web_sm")

# 2. 定义文章文本
article_text = """
Apple Inc. is a multinational technology company that designs and manufactures consumer electronics and software.
Founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976, Apple has become one of the largest companies 
in the world. Its most famous products include the iPhone, iPad, and Mac computers. Tim Cook is the CEO of Apple.
"""

# 3. 使用 spaCy 提取实体和关系
doc = nlp(article_text)

print(doc)