
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema.runnable import RunnableMap, RunnableBranch, RunnableLambda
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.globals import set_debug

# set_debug(True)


class TravelQASystem:
    '''
    ===========================================================================
    1. 初始化：LLM、嵌入模型、工具、知识库存储
    ===========================================================================
    '''
    def __init__(self, openai_api_key, tavily_key, embed_path):
        # 初始化语言模型
        self.llm = ChatOpenAI(api_key=openai_api_key,
                              base_url=os.getenv("DASHSCOPE_BASE_URL"),
                              model="qwen3-235b-a22b-instruct-2507")

        # 初始化搜索工具
        self.search_tool = TavilySearchResults(tavily_api_key=tavily_key)

        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_path)

        # 构建景点知识库
        self.attraction_data = [
            "公司写字楼：室内，人少，凉快，环境一般，可长时间呆，没有吃的，但是平日经常去，24h开启",
            "楼下公园：室外，小巧，人流量一般，环境好，有吃的，6:00-21:00",
            "万达广场：室内，较远需要地铁，人多，环境差，有大量吃的，12:00-23:00"
        ]

        # 使用内存型向量存储类
        self.vector_store = InMemoryVectorStore.from_texts(
            self.attraction_data, self.embeddings, k=2
        )

    '''
    ===========================================================================
    2. 创建 chain
    对行程进行建议，如果需要天气信息就调用 tool 查询
        1. 首先需要判断需不需要天气信息
    ===========================================================================
    '''
    def setup_runnable_pipeline(self):
        strOutputParser = StrOutputParser()
        # 1. 识别地点与查询类型
        parse_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content='请判断用户提出的需求中，天气因素是否占据很大的比重，并只返回 True/False'),
            ("user", "问题：{user_question}")
        ])
        parse_module = parse_prompt | self.llm | strOutputParser


        # 2. 并行获取数据:  天气 + 地点检索
        weather_query = RunnableLambda(
            lambda x: self.search_tool.invoke(f"周末北京天气")
        )
        attraction_retrieval = self.vector_store.as_retriever() | (lambda x: x[0].page_content)
        # 创建并行的  Runnable对象
        data_acquisition = RunnableMap({
            "weather": weather_query,
            "attraction": attraction_retrieval,
        })


        # 3. 回答
        generate_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是行程安排助手，需从用户问题中提取用户需求，并且在提供的可选列表中选择适合的地点"),
            ("user", """地点信息：{attraction}
                        天气情况：{weather}
                        请生成1条行程建议，包含注意事项（如天气相关准备）""")
        ])
        generate_module = generate_prompt | self.llm | strOutputParser


        # 4. 整合
        self.qa_pipeline = (
            # 阶段1：解析问题
            parse_module
            # 阶段2：并行获取数据（仅当查询类型为天气或行程时触发）   RunnableBranch判断
            | RunnableBranch(
                (lambda x: x == "True", data_acquisition),
                lambda x: { "weather":"未知", "attraction": attraction_retrieval.invoke(x) }
            )
            # 阶段3：生成回答
            | generate_module
        )

    '''
    ===========================================================================
    3. 回答提问：pipeline.invoke(question)
    ===========================================================================
    '''
    def process_user_question(self, user_question):
        input_data = {"user_question": user_question}
        response = self.qa_pipeline.invoke(input_data)
        return response



if __name__ == '__main__':
    # 需求   根据用户的需求  查询天气和景点的建议
    tqs = TravelQASystem(os.getenv("DASHSCOPE_API_KEY"), os.getenv("TAVILY_API_KEY"), r'C:\dev\llm\local-model\bge-large-zh-v1.5')
    tqs.setup_runnable_pipeline()
    ques1 = '万达广场几点开门'
    ans = tqs.process_user_question(ques1)
    print('问题:{}\n回答:{}'.format(ques1, ans))