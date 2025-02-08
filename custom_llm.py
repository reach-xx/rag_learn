from typing import Any
from typing import ClassVar
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

class MyLLM(CustomLLM):
    model_name: str = "custom"
    dummy_response: ClassVar[str] = "你好，我是一个正在开发中的大模型......"

#实现metadata接口
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
        )

    #实现complete接口
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.dummy_response)

    #实现stream_complete接口
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)

llm = MyLLM()
# resp = llm.stream_complete('你好！')
# for r in resp:
#     print(r)
resp = llm.complete('你好！')
print(resp)