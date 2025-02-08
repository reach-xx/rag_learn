import ollama, sys, chromadb
from load import getconfig

# 嵌入模型与大模型
embedmodel = getconfig()["embedmodel"]
llmmodel = getconfig()["mainmodel"]

while True:
    query = input("Enter your query: ")
    if query.lower() == 'quit':
        break
    else:
        # 生成Prompt
        modelquery = f"""
                    你是一个智能对话机器人： 用户发送一个消息，你能够提取文字中的关键实体，并能够说明其意图，如果不是下面指定意图可以回复不知道，输出格式
                    按照json格式输入，如下：
                    
                    ###########################
                    例子1：
                    输入:  帮我查下五山校区博学楼4层405教室的上课情况
                    输出： 
                    ##json
                    {
                        "action": 'inspection',
                        "address": {"campus":'五山校区', "building":'博学楼', "floor":'4层',"room":'405'}
                    }
                    
                    例子2：
                    输入:  帮我查下五山校区博学楼的上课情况
                    输出： 
                    ##json
                    {
                        "action": "inspection",
                        "address": {"campus":"五山校区", "building":"博学楼", "floor":"None","room":"None"}
                    }
                    
                    
                    例子3：
                    输入:  帮我查下大学城校区A2教学楼3层308教室的上课情况
                    输出： 
                    ##json
                    {
                        "action": "inspection",
                        "address": {"campus":"五山校区", "building":"A2教学楼", "floor":"3层","room":"308"}
                    }
                    
                    例子4：
                    输入:  帮我查下今天杨邵华老师的课程情况
                    输出： 
                    ##json
                    {
                        "action": "search_course",
                        "teacher": "杨邵华",
                        "date": "今天"	
                    }
                                 
                    例子5：
                    输入:  帮我查下1月13日任凯琳老师的课程情况
                    输出： 
                    ##json
                    {
                        "action": "search_teacher",
                        "teacher": "任凯琳",
                        "date": "1月13日"	
                    }
                            
                    
                    例子6：
                    输入:  帮我查下人工智能的课程信息
                    输出： 
                    ##json
                    {
                        "action": "search_course",
                        "course": "人工智能"
                    }
  
                    例子7：
                    输入:  帮我查下高分子材料与工程的课程信息
                    输出： 
                    ##json
                    {
                        "action": "search_course",
                        "course": "高分子材料与工程"
                    }
                      
                    例子7：
                    输入:  今天的天气怎么样？
                    输出： 
                    ##json
                    {
                        "action": "no_know",
                    }
                      
                    ##################
                    我的问题： {query}
        """

        # 交给大模型进行生成
        stream = ollama.generate(model=llmmodel, prompt=modelquery, stream=True)
        # 流式输出生成的结果
        for chunk in stream:
            if chunk["response"]:
                print(chunk['response'], end='', flush=True)

