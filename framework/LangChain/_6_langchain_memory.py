from langchain_community.chat_message_histories import ChatMessageHistory
import json
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# MessagesPlaceholder 占位    在提示词占位
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
# 本地存储
from langchain.schema import messages_from_dict, messages_to_dict
import os
'''
===========================================================================
1. Chat Messages
===========================================================================
'''
history = ChatMessageHistory()
print(history)
history.add_user_message("hi!")
history.add_user_message("你好")
history.add_ai_message("whats up?")
print(history.messages)

'''
===========================================================================
2. `RunnableWithMessageHistory` 
===========================================================================
'''

# 初始化大语言模型（通义千问）
llm = ChatOpenAI(
    api_key=os.getenv("api_key"),  # 从环境变量读取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容端点
    model="qwen-turbo"  # 使用qwen-turbo模型
)

# 创建对话提示模板
prompt = ChatPromptTemplate.from_messages([
    # 系统角色设定
    ("system", "你是一个友好的助手"),
    # 历史消息占位符（变量名必须与链配置中的history_messages_key一致）
    MessagesPlaceholder(variable_name="history", optional=True),
    # 用户输入占位符
    ("user", "{input}")
])

# 构建基础对话链（组合提示模板和语言模型）
base_chain = prompt | llm

# 创建一个回话存储的字典
store = {}

# 作用  用来判断是新用户 还是老用户
def get_session_history(session_id):
    # 判断是那个用的消息记录
    """获取或创建会话历史存储对象
    Args:
        session_id: 会话唯一标识（用于多会话隔离）
    Returns:
        对应会话的聊天历史记录对象
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # 初始化空历史记录
    return store[session_id]


# 创建支持历史记录的对话链   不是存储
conversation = RunnableWithMessageHistory(
    base_chain,  # 基础对话链
    get_session_history=get_session_history,  # 从 store 中获取
    input_messages_key="input",  # 输入文本的键名
    history_messages_key="history"  # 历史记录的键名（需与提示模板中的变量名一致）
)


# 把消息存在文件当中
def save_memory(filepath, session_id):
    """保存指定会话的历史记录到文件
    Args:
        filepath: 文件保存路径（建议使用.json扩展名）
        session_id: 要保存的会话ID（默认"default"）
    """
    history = get_session_history(session_id)
    # 将消息对象列表转换为字典格式
    dicts = messages_to_dict(history.messages)
    # 写入JSON文件（UTF-8编码）
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(dicts, f, ensure_ascii=False)


# 重文件中读取消息
def load_memory(filepath, session_id):
    """从文件加载历史记录到指定会话
    Args:
        filepath: 历史记录文件路径
        session_id: 要加载到的会话ID（默认"default"）
    """
    with open(filepath, "r", encoding='utf-8') as f:
        dicts = json.load(f)
    # 将字典转换回消息对象列表
    messages = messages_from_dict(dicts)
    # 更新全局存储中的会话历史  恢复聊天消息
    store[session_id] = ChatMessageHistory(messages=messages)


# 进行提问
def legacy_predict(input_text, session_id):
    """
    Args:
        input_text: 用户输入文本
        session_id: 会话ID（默认"default"）
    Returns:
        AI生成的回复文本
    """
    return conversation.invoke(
        {"input": input_text},  # 输入参数
        # 配置参数（必须包含session_id来关联历史记录）
        config={"configurable": {"session_id": session_id}}
    ).content


if __name__ == '__main__':
    SESSION_ID = "default"  # 会话ID
    # # 模拟连续对话（4轮）
    # legacy_predict("你好", SESSION_ID)  # 问候
    # legacy_predict("你是谁", SESSION_ID)  # 身份确认
    # legacy_predict("你的背后实现原理是什么", SESSION_ID)  # 技术原理询问
    
    # # 查询对话历史（第4轮）
    # last_response = legacy_predict('截止到现在我们聊了什么?', SESSION_ID)
    # print("最后一次回答:", last_response)
    
    # # 持久化保存对话历史（JSON格式）
    # save_memory("./memory_new.json", SESSION_ID)

    # 模拟重新加载历史记录（清空当前会话后重新加载）
    load_memory("./framework/LangChain/data/memory_new.json", SESSION_ID)
    # 验证历史恢复效果（第5轮）
    reload_response = legacy_predict("我回来了，我们之前都聊了一些什么?", SESSION_ID)
    print("\n恢复后的回答:", reload_response)
