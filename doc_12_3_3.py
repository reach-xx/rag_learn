import os
# from multiprocessing.managers import BaseManager
# from werkzeug.utils import secure_filename
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from multiprocessing.managers import BaseManager
import uvicorn

app = FastAPI()

# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 连接Index Server模块
manager = BaseManager(('192.168.97.215', 5602), b'123456')
manager.register('query_index')
manager.register('insert_into_index')
manager.register('get_documents_list')
manager.connect()

@app.get("/")
def home():
    return "Hello, World!"

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(content="Invalid request parameters", status_code=400)


@app.get("/query/")
def query_index(request: Request, query_text: str):
    global manager
    if query_text is None:
        return JSONResponse(content="No text found, please include a ?text=blah parameter in the URL", status_code=400)

    response = manager.query_index(query_text)._getvalue()
    response_json = {
        "text": str(response),
        "sources": [{"text": str(x.text),
                     "similarity": round(x.score, 2),
                     "doc_id": str(x.id_)
                     } for x in response.source_nodes]
    }
    return JSONResponse(content=response_json, status_code=200)

@app.post("/uploadFile")
async def upload_file(request: Request, file: UploadFile = File(...), filename_as_doc_id: bool = False):
    global manager
    try:
        contents = await file.read()
        print(f"contents: {contents}")
        filepath = os.path.join('data', file.filename)
        with open(filepath, "wb") as f:
            f.write(contents)

        if filename_as_doc_id:
            manager.insert_into_index(filepath, doc_id=file.filename)
        else:
            manager.insert_into_index(filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return JSONResponse(content="Error: {}".format(str(e)), status_code=500)

    if os.path.exists(filepath):
        os.remove(filepath)

    return JSONResponse(content="File inserted!", status_code=200)

@app.get("/getDocuments")
def get_documents(request: Request):
    document_list = manager.get_documents_list()._getvalue()
    return JSONResponse(content=document_list, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5601)

