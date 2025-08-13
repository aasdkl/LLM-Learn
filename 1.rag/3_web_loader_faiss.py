import os

from langchain_community.document_loaders import WebBaseLoader
import bs4
# 对于嵌入模型，这里通过 API调用  阿里社区提供的向量模型库
from langchain_community.embeddings import DashScopeEmbeddings
# 使用此嵌入模型将文档摄取到矢量存储中
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

'''
使用 FAISS 爬取、分割、向量化、查询
    WebBaseLoader
    FAISS.from_documents
'''

def faiss_conn():
    # 1. 爬取网页
    loader = WebBaseLoader(
        web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm", # 民法典
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))       # 主要内容在 #UCAP-CONTENT 节点
    )
    docs = loader.load()
    # print(docs)

    # 2. 分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs) # [Document]
    # print(documents)

    # 3. 向量存储
    # 这里使用阿里提供的 langchain 接口，https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2587654
    embeddings = DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), model='text-embedding-v2')
    vector = FAISS.from_documents(documents, embeddings) #embeddings 会将 documents 中的每个文本片段转换为向量，并将这些向量存储在 FAISS 向量数据库中
    return vector

vector=faiss_conn()
print(vector.similarity_search("请问14岁犯罪会被释放吗", 3))
# similarity_search_with_score 带有分数的