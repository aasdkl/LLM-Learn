from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
import os

llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"),
                 model='qwen-plus',
                 temperature=0)

def invoke_demo():
    template = "桌上有{number}个苹果，四个桃子和 3 本书，一共有几个水果?直接回答答案"
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    # 管道形成了一个RunnableSequence。这实际上相当于一个简单的链，用于替代 LLMChain


    # 【invoke】 输入为字典格式
    result = chain.invoke({"number": "3"})
    print(f"【invoke】: {result.content}")



    # 【batch】
    input_list = [{"number": "3"}, {"number": "10"}]
    result = chain.batch(input_list)
    print(f"【batch1】: {result[0].content}")
    print(f"【batch2】: {result[1].content}")



    # 【predict】 指定关键字参数，不能直接用在链上（较旧的 API，可能被弃用）
    from langchain.chains.llm import LLMChain
    chain_old = LLMChain(llm=llm, prompt=prompt)
    result = chain_old.predict(number=3)
    print(f"【predict】: {result}")

def chain_demo():
    # 基本的 LLMChain 被管道符代替

    # 数学链，转换为可以使用 Python 的 numexpr 库执行的表达式
    llm_math = LLMMathChain.from_llm(llm)
    res = llm_math.invoke("5 ** 3 + 100 / 2的结果是多少？")
    print(res)



    # SQL 链，将自然语言转换成数据库的SQL查询

    # 连接 MySQL 数据库
    uri = f"postgresql+psycopg2://kabAdmin:P%40ssw0rd!@kintonetooldb.postgres.database.azure.com/dev_v2"
    db = SQLDatabase.from_uri(uri)
    chain = create_sql_query_chain(llm=llm, db=db)
    # print(chain.get_prompts()[0].pretty_print())

    response = chain.invoke({"question": "今年创建了哪些用户，只要输出符合条件的所有数量，并且只要输出 sql 不要其余文档结构", "table_names_to_use": ["user"]})
    print(response)
    print("查询结果：", db.run(response))

    # chain = create_sql_query_chain(llm=llm, db=db) | (lambda x: x.replace("SQLQuery: ", "")) | QuerySQLDatabaseTool(db=db)
    # response = chain.invoke({"question": "今年有哪些用户登陆过，只要输出符合条件的所有数量，并且只要输出 sql 不要其余文档结构", "table_names_to_use": ["user"]})
    # print(response)


invoke_demo()
chain_demo()