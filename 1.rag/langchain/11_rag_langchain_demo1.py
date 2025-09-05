
# pip install langchain langchain-chroma "unstructured[all-docs]" pydantic lxml langchainhub pi_heif
# - poppler
#   - 下载地址：https://github.com/oschwartz10612/poppler-windows/releases
#   - 导入方式：解压缩后将bin文件夹路径导入windows系统‘Path’变量，可在命令行验证是否成功
#   - 验证方式：终端命令`pdfinfo -v`
# - Tesseract
#   - 下载地址：https://digi.bib.uni-mannheim.de/tesseract/
#   - 导入方式：下载后直接安装即可，安装完成后查看是否安装成功，若未识别，手动添加安装根目录至系统‘Path’变量
#   - 验证方式：终端命令`tesseract -v`

from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

embedding_model_path = r'C:\dev\llm\local-model\bge-large-zh-v1.5'
# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)

# 定义文件路径（需根据自己的路径修改）
path = os.path.join(os.path.dirname(__file__), "./data/厦门灿坤实业股份有限公司_2019年度报告.pdf")



# ------------------------ 第一阶段：PDF解析处理 ------------------------
# 使用unstructured库解析PDF文档
raw_pdf_elements = partition_pdf(
    filename=path,
    extract_images_in_pdf=False,  # 不提取PDF中的图片
    infer_table_structure=True,  # 启用表格结构识别
    max_characters=4000,  # 每个文本块最大字符数
    new_after_n_chars=3800,  # 达到3800个字符后分新块    阈值
    combine_text_under_n_chars=2000,  # 合并小于2000个字符的碎片文本
    strategy='hi_res',
)

print(raw_pdf_elements)
# 统计各类元素数量
category_counts = {}
for element in raw_pdf_elements:
    print(element)
    category = str(type(element))
    category_counts[category] = category_counts.get(category, 0) + 1

print("元素类型统计:", category_counts)


# ------------------------ 第二阶段：元素分类处理 ------------------------
# 分类处理PDF元素 把文本和表格元素分类存在列表
# 表格数据信息
table_elements = []
# 文本数据信息
text_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        table_elements.append(str(element))
    elif "unstructured.documents.elements.Text" in str(type(element)):
        text_elements.append(str(element))
    elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
        text_elements.append(str(element))

# 手动将文本内容分块
chuck_text_elements = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200).split_text(''.join(text_elements))
print(f'文本块内容：{chuck_text_elements}')
print(f"识别到表格数量: {len(table_elements)}, 文本块数量: {len(chuck_text_elements)}")
print("表格示例:", table_elements[0:10])



# ------------------------ 第三阶段：内容摘要生成 ------------------------
# 定义摘要生成提示模板
prompt_text = """您是一个专业的内容摘要助手，请对以下表格或文本块进行简洁的总结：
{element}"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# 初始化大模型（此处使用阿里云通义千问）
model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)


# 构建摘要生成链
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# 批量生成表格摘要
table_summaries = summarize_chain.batch(table_elements, {"max_concurrency": 5})  # 并发处理
print("表格摘要示例:", table_summaries[0:10])

# 批量生成文本摘要
text_summaries = summarize_chain.batch(chuck_text_elements, {"max_concurrency": 5})
print("文本摘要示例:", text_summaries[0:10])

# ------------------------ 第四阶段：构建多向量检索器 ------------------------

# 创建向量数据库（用于存储摘要）
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=embeddings_model
)

# 创建内存存储（用于存储原始内容）
store = InMemoryStore()
id_key = "doc_id"  # 文档标识键

# 初始化多向量检索器  多用检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# 添加文本数据到检索器
text_ids = [str(uuid.uuid4()) for _ in chuck_text_elements]
# 把摘要之后文本的内容转换成Document对象
summary_texts = [Document(page_content=s, metadata={id_key: text_ids[i]}) for i, s in enumerate(text_summaries)]
# 把Document对象添加到检索器中
retriever.vectorstore.add_documents(summary_texts)
# 把文本数据和摘要的内容 通过id进行绑定
retriever.docstore.mset(list(zip(text_ids, chuck_text_elements)))


# 添加表格数据到检索器   给每个表格数据生成一个id
table_ids = [str(uuid.uuid4()) for _ in table_elements]
# 循环取出摘要之后的结果  生成Document对象 metadata存id数据   用来绑定源表格
summary_tables = [Document(page_content=s, metadata={id_key: table_ids[i]})for i, s in enumerate(table_summaries)]
# 添加数据到向量数据库
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, table_elements)))


# ------------------------ 第五阶段：构建问答链 ------------------------
# 定义问答提示模板
template = """请仅根据以下上下文（包含文本和表格）回答问题：
{context}
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建问答链
# RunnablePassthrough   透传   虚拟环境
chain = (
        {"context": lambda x: retriever.invoke(input=x["question"]), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

# 示例问答测试
question = "营业收入构有哪些?可以往哪里发展?"
print("回答：", chain.invoke({"question": question}))
print("检索结果：", retriever.invoke(question))




