import os
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,     # 等价于OpenAI接口中的 assistant role AI 模型的回复消息
    HumanMessage,  # 等价于OpenAI接口中的 user role      表示用户输入的消息
    SystemMessage  # 等价于OpenAI接口中的 system role    系统级指令或背景设定
)
'''
基本调用方式：
    llm.invoke()

    chain = prompt | llm
    chain.invoke()
'''

llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"),
                 model_name="qwen-turbo",
                 temperature=1.5)

''' 1. 静态方式 '''
# response = llm.invoke("什么是大模型？") # 直接提供问题，一轮回答
messages = [
    SystemMessage(content="你是各位老师的个人助理。你叫小戈"),
    HumanMessage(content="我的名字叫小张, 今天天气怎么样"),
    AIMessage(content="小张同学你好，不好意思，暂时无法获取天气情况"),  # 将 AI 的回复一起返回，增加上下文记忆
    HumanMessage(content="什么？"),
]
response = llm.invoke(messages)
print(f'【1.静态方式】：{response.content}')
print("=" * 100)



'''2. 模板方式 ChatPromptTemplate（批量处理相似任务，底层依然是使用三类 Message）'''
# 1. LLM提示模板 PromptTemplate：常用的String提示模板
# 2. 聊天提示模板 ChatPromptTemplate： 常用的Chat提示模板，用于组合各种角色的消息模板，传入聊天模型。消息模板包括：ChatMessagePromptTemplate、HumanMessagePromptTemplate、AIlMessagePromptTemplate、SystemMessagePromptTemplate等
# 3. 样本提示模板 FewShotPromptTemplate：通过示例来教模型如何回答
# 4. 部分格式化提示模板：提示模板传入所需值的子集，以创建仅期望剩余值子集的新提示模板。
# 5. 管道提示模板 PipelinePrompt： 用于把几个提示组合在一起使用。
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 实现和【1. 静态方式】一样的内容
# doc_template = ChatPromptTemplate.from_messages([
#     ("system", "你是各位老师的个人助理。你叫小戈"),
#     ("user", "我的名字叫小张, 今天天气怎么样"),
#     ("ai", "小张同学你好，不好意思，暂时无法获取天气情况"),
#     ("user", "{input}"),
# ])
# chain = doc_template | llm
# response = chain.invoke({"input": "什么？"})
# print(f'【2.模板方式】：{response.content}')
# print("=" * 100)

doc_template = ChatPromptTemplate.from_messages([
    # 可以嵌套使用 SystemMessagePromptTemplate/HumanMessagePromptTemplate，也可以直接传元组
    SystemMessagePromptTemplate.from_template("作为{language}技术专家，使用{style}风格，一句话简短回答"),
    # 底层使用 prompt.format('xxx') 的方式
    ("user", "提问：\n{question}")
])

chain = doc_template | llm
response = chain.invoke({"language": "Java", "style": "通俗", "question": "JVM 是什么？"})
print(f'【2.模板方式】：{response.content}')
print("=" * 100)



'''3. 限制上下文记忆'''
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

# 这里仅仅是演示，建议使用 LangGraph https://python.langchain.com/docs/versions/migrating_memory/
memory = ConversationBufferWindowMemory(
    k=1,                        # 保留最近1轮对话
    memory_key="chat_history",  # 在prompt中使用的变量名
    return_messages=True        # 返回Message对象而非字符串
)
memory.save_context({"input": "杭州特色菜"}, {"output": "东坡肉、叫化鸡、小炒肉"})
memory.save_context({"input": "推荐清淡的"}, {"output": "龙井虾仁、西湖莼菜汤"})

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是美食顾问小戈"),           # 封装的 message
    MessagesPlaceholder(variable_name="chat_history"),  # 占位符，用于插入历史记录
    ("human", "{user_input}")                           # 新用户输入
])

input_dict = {
    "user_input": "你刚刚推荐的东西里面有辣的吗？",
    "chat_history": memory.load_memory_variables({})["chat_history"]
}

# final_prompt = prompt.format_messages(**input_dict) # 只是输出 prompt，和实际使用 chain.invoke 的时候一致
chain = prompt | llm
response = chain.invoke(input_dict)
print(f'【3.限制上下文记忆】：{response.content}')