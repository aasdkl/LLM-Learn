

from langchain_chroma import Chroma
from 元数据 import docs, metadata_field_info
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os


# 本地embedding模型地址
embedding_model_path = r'C:\dev\llm\local-model\bge-large-zh-v1.5'
# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)

# 文档内容描述（指导LLM理解文档内容）
document_content_description = "Brief description of technical articles"

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)
# 创建向量数据库
vectorstore = Chroma.from_documents(docs, embeddings_model)

# 创建自查询检索器（核心组件）
"""
请帮我把当前这个问题进行转换 


SelfQueryRetriever.from_llm解析步骤:
1. 用户的问题("2023年评分超过8分的机器学习论文")
    LLM 任务(会由底层源码自己实现)：
    识别用户意图中的显式条件（如“2023年”对应 year=2023）。
    提取隐式语义关键词（如“机器学习论文”作为向量搜索关键词）。
    将模糊表述转换为结构化操作符（如“超过8分” → {"$gt": 8}）。
    {
        "query": "机器学习论文",
        "filter": {"year": 2023, "rating": {"$gt": 8}}
    }

2. 元数据过滤
    根据 filter 对向量数据库中的文档元数据进行筛选。(筛选出 year=2023 且 rating>8 的所有文档。)

3. 语义搜索
    在元数据过滤后的文档子集中，用 query 进行向量相似性搜索。

4. 结果合并与排序
    综合元数据匹配度（如完全匹配 year=2023）和语义相关性（向量得分），返回排序后的文档列表

"""

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True #严格限制
)
# print(retriever.invoke("作者B在2024年发布的文章"))
# 根据限定条件进行设置 enable_limit=True
print(retriever.invoke("我想了解一篇评分在9分以上的文章"))




