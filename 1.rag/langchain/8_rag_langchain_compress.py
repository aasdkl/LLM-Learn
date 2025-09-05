


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, LLMChainFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
import os


# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# 加载文档
documents = TextLoader(os.path.join(os.path.dirname(__file__), "./data/deepseek介绍.txt"), encoding="utf-8").load()
# 文档分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

#　向量模型　　
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("api_key"),
)
# 存数据到向量数据库
retriever = Chroma.from_documents(texts, embeddings_model).as_retriever()

# print("-------------------压缩前--------------------------")
# docs = retriever.invoke("deepseek的发展历程")
# pretty_print_docs(docs)


llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)

# LLMChainExtractor 具体执行文档内容精炼的压缩器, 通过 LLM 对文档进行精炼
compressor = LLMChainExtractor.from_llm(llm)
# ContextualCompressionRetriever 将基础检索器和 压缩器结合
# 处理的逻辑   把检索到的数据 给到压缩器进行处理
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )
# compression_data = compression_retriever.invoke("deepseek的发展历程")
# print("-------------------LLMChainExtractor压缩后--------------------------")
# pretty_print_docs(compression_data)



# 不会修改源文档内容
_filter = LLMChainFilter.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=_filter, base_retriever=retriever
# )
# print("-------------------LLMChainFilter压缩后--------------------------")
# compressed_docs = compression_retriever.invoke("deepseek的发展历程")
# pretty_print_docs(compressed_docs)



# EmbeddingsFilter 具体执行文档内容过滤的压缩器, 通过检索到的文档相识度进行过滤
embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.66)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=embeddings_filter, base_retriever=retriever
# )
# compressed_docs = compression_retriever.invoke("deepseek的发展历程")
# print("-------------------EmbeddingsFilter压缩后--------------------------")
# pretty_print_docs(compressed_docs)



# 综合使用
# 1. 首先`TextSplitters`可以用作文档转换器，将文档分割成更小的块
# 2. 然后`EmbeddingsRedundantFilter` 根据文档之间嵌入的相似性来过滤掉冗余文档
# 3. 最后 `EmbeddingsFilter` 根据与查询的相关性进行过滤。 

# CharacterTextSplitter 固定文本分块
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="。 ")
# EmbeddingsRedundantFilter是文档和文档之间进行过滤的压缩器
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)
# EmbeddingsFilter是查询问题和文档的相似度进行过滤
relevant_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.66)

# DocumentCompressorPipeline 是一个用于串联多个文档处理步骤的组件，其核心作用是通过组合不同的文档转换/过滤工具，构建一个多阶段的文档压缩流水线
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)
print("-------------------DocumentCompressorPipeline压缩后--------------------------")
compressed_docs = compression_retriever.invoke("deepseek的发展历程")
pretty_print_docs(compressed_docs)


