# 加载与读取文档
import mimetypes
import os, configparser


def loadtext(path):
    path = path.rstrip()
    path = path.replace(' \n', '')

    # 转换绝对路径
    filename = os.path.abspath(path)

    # 判断文档存在，并获得文档类型

    filetype = ''
    if os.path.isfile(filename):
        filetype = mimetypes.guess_type(filename)[0]
    else:
        print(f"File {filename} not found")
        return None

    # 读取文档内容
    text = ""
    if filetype != 'text/plain':
        return None
    else:
        with open(filename, 'rb') as f:
            text = f.read().decode('utf-8')

    return text


# 这里配置了一个简单的配置器，用于读取模型名称的配置，后面要用
def getconfig():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return dict(config.items("main"))
