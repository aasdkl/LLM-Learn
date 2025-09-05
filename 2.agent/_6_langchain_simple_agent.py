
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import initialize_agent, create_openai_functions_agent, AgentType, AgentExecutor
from langchain_core.tools import Tool
from langchain.tools.retriever import create_retriever_tool
import os
import sys
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
# 将项目根目录添加到系统路径
sys.path.append(project_root)

from framework.LangChain._2_langchain_loader import faiss_conn

'''
agent

1. 首先将文档读取到数据库中（RAG），设为 retriever
2. create_retriever_tool 创建工具，需要详细说明工具的目的
3. 创建 ChatPromptTemplate，必须要有：
    - {{tools}}             会自动添加（需要双大括号表示由langchain自动生成）
    - {input}               用户输入
    - variable_name="agent_scratchpad"  思考过程
    并且按照顺序：系统指令（tools）-> 对话历史（chat_history）-> 当前用户输入（input）-> 思考过程（agent_scratchpad）
4. 创建 create_openai_functions_agent

* LangSmith 有在线的提示词模板，需要 LANG_SMITH_API
'''

llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"),
                 model='qwen-plus',
                 temperature=0)

# 1. 读取数据（创建出 faiss 检索器）
retriever = faiss_conn().as_retriever()

# 2. 创建RAG工具（基于检索器）
retriever_tool = create_retriever_tool(
    retriever,
    "中华人民共和国民法典的一个检索器工具",
    "搜索有关中华人民共和国民法典的信息。关于中华人民共和国民法典的任何问题，您必须使用此工具!",
)

@tool
def query_order_status(order_id: str) -> str:
    """根据订单ID查询订单状态。输入order_id必须是数字。"""
    return '已发货' if int(order_id.strip()) % 2 == 0 else '未发货'

tools = [
    # RAG
    retriever_tool,
    # 搜索
    TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY")),
    # 订单查询工具
    query_order_status
]


agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS, # 使用OPENAI_FUNCTIONS，它基于OpenAI的Function Calling功能，能更好地处理JSON参数。
    verbose=True,
    handle_parsing_errors=True, # 至关重要！让Agent执行器具备一定的错误处理能力，在解析失败时尝试修复或继续。
    max_iterations=10, # 明确设置最大迭代次数，防止无限循环
    # 对于OPENAI_FUNCTIONS，可以通过agent_kwargs自定义系统消息的一部分
    agent_kwargs={
        "system_message": "你是一个有用的助手。请严格遵守工具调用规范：必须提供完整且有效的JSON参数。"
    }
)


# 3. prompt
# https://smith.langchain.com/hub
# 使用在线的提示词模板，需要 LANG_SMITH_API
# prompt = hub.pull("hwchase17/openai-functions-agent", include_model=True, api_key=os.getenv("LANG_SMITH_API"))
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """你是一个有用的助手。请严格遵守工具调用规范：必须提供完整且有效的JSON参数。"""),
#     # MessagesPlaceholder(variable_name="chat_history", optional=True),
#     ("human", "{input}"), 
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

# 创建agent
# agent = create_openai_functions_agent(llm, tools, prompt)
# verbose 详细模式，  # 启用自动处理解析错误
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10, handle_parsing_errors=True)

# 定义一些测试询问
queries = [
    "请问订单1024的状态是什么？",
    "2024年谁胜出了美国总统的选举",
    "请问民法典中，14岁犯罪会被释放吗？",
]

responses = agent_executor.batch([{"input": input} for input in queries])

# 运行代理并输出结果
for response in responses:
    print(f"代理回答：{response}\n")

