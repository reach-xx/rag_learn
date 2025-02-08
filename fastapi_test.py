from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, FastAPI, HTTPException,status
import uvicorn
import time

app = FastAPI()
# chat_router = r = APIRouter()
# 生成器函数，返回符合 SSE 格式的数据
def event_generator():
    # 模拟逐步生成数据
    yield "data: Hello, world!\n\n"
    time.sleep(1)  # 模拟延迟
    yield "data: This is another message\n\n"
    time.sleep(1)
    yield "data: Stream has ended\n\n"

# FastAPI 路由返回流式响应
@app.post("/api/chat")
async def chat():
    return StreamingResponse(event_generator(), media_type="text/event-stream")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

#将chat_router路由包含进来，有利于路由的模块化组织管理
# app.include_router(chat_router, prefix="/api/chat")

if __name__ == "__main__":
    uvicorn.run(app="fastapi_test:app", host="0.0.0.0", port=9090, reload=True)