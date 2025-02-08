from llama_index.readers.file import ImageReader
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.schema import Document,MetadataMode
from llama_index.readers.web import BeautifulSoupWebReader
from typing import Any,Tuple,Dict

#===================加载图片ImageReader=======================
# image_reader = ImageReader(keep_image = True)
# reader = SimpleDirectoryReader(input_files=["./data/frame.jpg"], file_extractor={".jpg":image_reader})
# print(reader.load_data()[0].image)

#定义一个打印Document数组的方法，这个方法在后面经常使用
def print_docs(docs:list[Document]):
    print('Count of documents:',len(docs))
    for index,doc in enumerate(docs):
        print("-----")
        print(f"Document {index}")
        print(doc.get_content(metadata_mode=MetadataMode.ALL))
        print("-----")


#=================web网站上加载内容=============================

web_loader = SimpleWebPageReader(html_to_text=True)
docs = \
web_loader.load_data(urls=["https://cloud.baidu.com/doc/COMATE/s/rlnvnio4a"])
# print_docs(docs)         #自定义的打印docs变量的方法



#定义一个个性化的网页内容提取方式
def _baidu_reader(soup: Any, url: str, include_url_in_text: bool = True) ->Tuple[str, Dict[str, Any]]:
    main_content = soup.find(class_='main')
    if main_content:
       text = main_content.get_text()
    else:
       text = ''
    return text, {"title": soup.find(class_="post__title").get_text()}

web_loader = \
BeautifulSoupWebReader(website_extractor={"cloud.bai**.com":_baidu_reader})

docs = \
web_loader.load_data(urls=["https://cloud.baidu.com/doc/COMATE/s/rlnvnio4a"])
# print_docs(docs)

#========================加载数据库的数据=================================
from llama_index.readers.database import DatabaseReader
from llama_index.core.schema import Document,TextNode,MetadataMode
db = DatabaseReader(
    scheme="postgresql",  # Database Scheme
    host="localhost",  # Database Host
    port="5432",  # Database Port
    user="postgres",  # Database User
    password="123456",  # Database Password
    dbname="mydb",  # Database Name
)
docs = db.load_data(query="select * from questions")
print_docs(docs)