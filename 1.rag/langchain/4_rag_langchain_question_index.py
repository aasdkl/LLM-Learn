


from typing import List
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_huggingface import HuggingFaceEmbeddings
# 限定格式   web   前段数据校验   Field约束字段
from pydantic import BaseModel, Field
import uuid
import os

# 本地embedding模型地址
embedding_model_path = r'C:\dev\llm\local-model\bge-large-zh-v1.5'
# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)

# 初始化文档加载器列表
loader = TextLoader(os.path.join(os.path.dirname(__file__), "./data/deepseek介绍.txt"), encoding="utf-8")
docs = loader.load()

# 初始化递归文本分割器（设置块大小和重叠）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
# print(docs)

# 初始化llm
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)


class HypotheticalQuestions(BaseModel):
    """生成假设性问题"""
    # Field  ... 必传字段  问题必须是列表  对接受的数据进行效验限制
    questions: List[str] = Field(..., description="List of questions")


prompt = ChatPromptTemplate.from_template(
    """请基于以下文档生成3个假设性问题（必须使用JSON格式）:
    {doc}

    要求：
    1. 输出必须为合法JSON格式，包含questions字段
    2. questions字段的值是包含3个问题的数组
    3. 使用中文提问
    示例格式：
    {{
        "questions": ["问题1", "问题2", "问题3"]
    }}"""
)

# 创建假设性问题链
chain = (
        {"doc": lambda x: x.page_content}
        | prompt
        # 将LLM输出构建为字符串列表
        | llm.with_structured_output(HypotheticalQuestions)
        # 提取问题列表
        | (lambda x: x.questions)
)
# 在单个文档上调用链输出问题列表：
# print(chain.invoke(docs[0]))



# 批量处理所有文档生成假设性问题（最大并行数5）
hypothetical_questions = chain.batch(docs[:5], {"max_concurrency": 5})
print(hypothetical_questions)

# 初始化Chroma向量数据库（存储生成的问题向量）
vectorstore = Chroma(
    collection_name="hypo-questions", embedding_function=embeddings_model
)
# 初始化内存存储（存储原始文档）
store = InMemoryByteStore()

id_key = "doc_id"  # 文档标识键名

# 配置多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# 为每个原始文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 将生成的问题转换为带元数据的文档对象
question_docs = []
for (i, question_list) in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )

retriever.vectorstore.add_documents(question_docs)  # 将问题文档存入向量数据库
retriever.docstore.mset(list(zip(doc_ids, docs)))  # 将原始文档存入字节存储（通过ID关联）


# 执行相似性搜索测试
# query = "deepseek受到哪些攻击？"
# sub_docs = retriever.vectorstore.similarity_search(query)
# print(sub_docs)


prompt1 = ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}")

# 生成问题回答链
chain = RunnableMap({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt1 | llm | StrOutputParser()

query = "deepseek受到哪些攻击？"
# 生成问题回答
answer = chain.invoke({"question": query})
print("-------------回答--------------")
print(answer)

