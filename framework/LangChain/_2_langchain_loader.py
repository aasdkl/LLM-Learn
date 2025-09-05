import os

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, WebBaseLoader
import bs4
# 对于嵌入模型，这里通过 API调用  阿里社区提供的向量模型库
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

'''
不同 Loader 支持不同类型的数据源

- loader.load() 所有内容
- loader.load_and_split() 默认RecursiveCharacterTextSplitter分割

'''

'''
基本类型
'''
def basic_demo():
    '''
    ================================================================
    1. pdf
    ================================================================
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(current_dir, "data", "财务管理文档.pdf")
    loader = PyPDFLoader(path) # path 可以是网址
    pages = loader.load_and_split()  # 默认 RecursiveCharacterTextSplitter

    print("【1. pdf】")
    print(f"第0页：\n{pages[0]}")  ## 也可通过 pages[0].page_content只获取本页内容
    print("-" * 50)

    '''
    ================================================================
    2. doc
    ================================================================
    '''

    # 指定要加载的Word文档路径，无法按页分割，可以mode="elements"
    path = os.path.join(current_dir, "data", "人事管理流程.docx")
    loader = UnstructuredWordDocumentLoader(path, mode="elements")

    pages = loader.load()
    # pages = loader.load_and_split()
    print("【2. docx】")
    for i in range(0, 10):
        print(f"第{i}个元素：{pages[i].page_content}\t\t{pages[i].metadata['category']}")
    print("-" * 50)


'''
================================================================
3. WebBaseLoader
================================================================
对于数据库 FAISS/ChromaDB 都支持向量化（from_documents）、简单查询（similarity_search）
- 如果需要配合链式，必须先转为获取器 as_retriever
'''
def faiss_conn():
    # 1. 爬取网页 WebBaseLoader
    loader = WebBaseLoader(
        web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm", # 民法典
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))       # 主要内容在 #UCAP-CONTENT 节点
    )
    docs = loader.load()
    # print(docs)

    # 2. 分割文档 RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs) # [Document]
    # print(documents)

    # 3. 向量存储 from_documents
    # 这里使用阿里提供的 langchain 接口，https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2587654
    # embeddings = DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), model='text-embedding-v1')
    embeddings = HuggingFaceEmbeddings(model_name=r'C:\dev\llm\local-model\bge-large-zh-v1.5')
    vector = FAISS.from_documents(documents[5:8], embeddings) #embeddings 会将 documents 中的每个文本片段转换为向量，并将这些向量存储在 FAISS 向量数据库中
    return vector

if __name__ == '__main__':
    # basic_demo()
    vector = faiss_conn()
    print(vector.similarity_search("请问14岁犯罪会被释放吗", 3))
    # similarity_search_with_score 带有分数