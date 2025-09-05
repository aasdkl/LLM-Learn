

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# 本地embedding模型地址
embedding_model_path = r'C:\dev\llm\local-model\bge-large-zh-v1.5'

# 目标 URL
url = "https://news.pku.edu.cn/mtbdnew/15ac0b3e79244efa88b03a570cbcbcaa.htm"

# 加载网页
loader = WebBaseLoader(url)
docs = loader.load()
# print(docs)
# 查看长度
print(f"文章长度：{len(docs[0].page_content)}")


# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000) # chunk_overlap=200

# 创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)
# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=embeddings_model
)

# print(Document(page_content=docs[0].page_content.replace('\n', '').replace(' ', '')))

# 创建内存存储对象
store = InMemoryStore()
# 创建父文档检索器 子文档会存在向量数据库中, 夫文档会存在内存中
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 1}
)
# 添加文档集
retriever.add_documents([Document(page_content=docs[0].page_content.replace('\n', '').replace(' ', ''))])

# 查看分割之后的长度
print(len(list(store.yield_keys())))
print(len(vectorstore.get()['documents']))


# print("------------similarity_search------------------------")
# # 在向量数据库中搜索子文档
# sub_docs = vectorstore.similarity_search("天才AI少女是谁？")
# print(sub_docs)

# print("------------get_relevant_documents------------------------")
# # 过程 retriever.invoke先去检索子文档，根据子文档的元数据(metadata)找到父文档
# retrieved_docs = retriever.invoke("天才AI少女是谁？")
# print(retrieved_docs[0].page_content)



# 创建model
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)


# 创建chain
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("------------模型回复------------------------")

response = chain.invoke({"question": "天才AI少女是谁？"})
print(response)







