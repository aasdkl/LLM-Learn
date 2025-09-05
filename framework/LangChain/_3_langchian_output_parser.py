import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
'''
parser 主要是用来报错的
可以使用 parser.get_format_instructions() 得到一个 prompt
'''

llm = ChatOpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"),
                 model_name="qwen-turbo")

parser = JsonOutputParser()
# parser = CommaSeparatedListOutputParser()
format_instructions = parser.get_format_instructions()

print('-' * 50)
print(format_instructions)
print('-' * 50)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的程序员\n\n{format_instructions}"),
    ("user", "{input}")
])

chain = prompt | llm | parser

res = chain.invoke({"input": "langchain是什么?", "format_instructions": format_instructions})
print(res)