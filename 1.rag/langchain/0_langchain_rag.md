# 预检索优化

## 摘要索引 MultiVectorRetriever
1. 创建摘要(chain)
2. 使用 `MultiVectorRetriever`
   - 摘要放向量数据库，文档一般数据库
3. 在文档中需要有 `metadata: id_key`
4. 调用 `retriever.invoke`，自动得到父文档

```python
# 创建摘要生成链
chain = (
      {"doc": lambda x: x.page_content}
      | ChatPromptTemplate.from_template("总结下面的文档:\n\n{doc}")
      | llm
      | StrOutputParser()
)
summaries = chain.batch(docs, {"max_concurrency": 5})

# 检索器
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore = Chroma(...), # 摘要
    docstore = InMemoryStore(), # 文档
    id_key=id_key, # 必须有
)

# 摘要加入 id_key（每个 summary 生成 Document）
doc_ids = [str(uuid.uuid4()) for _ in docs]
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)
]
retriever.vectorstore.add_documents(summary_docs)

# 文档加入 id_key
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 查询
retrieved_docs = retriever.invoke("天才AI少女是谁？")
```



## 父子索引 ParentDocumentRetriever
1. 设置两个 splitter
2. 使用 `ParentDocumentRetriever`
   - 子文档放向量数据库，父文档一般数据库
   - 自动会生成对应的 id（先划出父文档，再在父文档划分子文档）
3. 调用 `retriever.invoke`，自动得到父文档

```python
# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# 创建子文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 检索器
retriever = ParentDocumentRetriever(
    vectorstore = Chroma(...), # 子文档
    docstore = InMemoryStore(), # 父文档
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 1},
    # ids=[...] # 不提供的话，会自动生成 uuid 作为连接
)

# 查询
retrieved_docs = retriever.invoke("天才AI少女是谁？")
```

## 假设性问题

- 和摘要索引几乎一样，一般不用

```python
# 创建假设性问题生成链
class HypotheticalQuestions(BaseModel):
    """生成假设性问题"""
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

chain = (
        {"doc": lambda x: x.page_content}
        | prompt
        # 将LLM输出构建为字符串列表
        | llm.with_structured_output(HypotheticalQuestions)
        # 提取问题列表
        | (lambda x: x.questions)
)
summaries = chain.batch(docs, {"max_concurrency": 5})

# 检索器同摘要

# 摘要加入 id_key（每个 q 生成 Document，因此是1对多）
doc_ids = [str(uuid.uuid4()) for _ in docs]
question_docs = []
for (i, question_list) in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )
retriever.vectorstore.add_documents(summary_docs)

# 文档加入 id_key
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 查询
retrieved_docs = retriever.invoke("天才AI少女是谁？")
```


## 元数据索引 SelfQueryRetriever

1. 预先准备元数据、文档内容描述(英文)
2. 文档中加上 metadata
3. 使用 `SelfQueryRetriever`
   - 需要开启 `enable_limit`
   - 底层会基于问题构造出结构化操作，之后对 metadata 操作

```python
# 元数据字段定义（指导LLM如何解析查询条件）
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="Technical domain of the article, options: ['AI', 'Blockchain', 'Cloud', 'Big Data']",
        type="string",
    ),
]

# 文档内容描述（指导LLM理解内容）
document_content_description = "Brief description of technical articles"

# 元数据
docs = [
    Document(
        page_content="作者A团队开发出基于深度学习的图像识别系统，在复杂场景下的识别准确率提升250%",
        metadata={"year": 2025, "rating": 9.3, "genre": "AI", "author": "A"},
    ),
]

# 检索器
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    doc_description, # 文档内容描述（指导LLM理解内容）
    metadata_field_info,
    enable_limit=True # 严格限制
)
# 根据限定条件进行设置 enable_limit=True
print(retriever.invoke("我想了解一篇评分在9分以上的文章"))
```


## 混合检索 EnsembleRetriever

1. 创建多个 retriever
   -  `db.as_retriever()` 和 `BM25Retriever`
2. 混合检索 `EnsembleRetriever`

```python
# 向量检索 split_docs
vector_retriever = vectorstore.as_retriever()

# 关键词检索 split_docs
BM25_retriever = BM25Retriever.from_documents(split_docs)

# 混合检索
retriever = EnsembleRetriever(retrievers=[BM25_retriever, vector_retriever], weights=[0.2, 0.8], k=3, search_kwargs={"k": 3})
retriever_doc = retriever.invoke(question)

```

# 查询优化 MultiQueryRetriever

1. `MultiQueryRetriever`


```python
# 打开日志，显示生成的问题
import logging
logging.basicConfig(level=logging.INFO)

# 生成 MultiQueryRetriever（使用默认模板生成问题）
retrieval_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    #prompt #使用默认模板生成问题
)
retrieval_from_llm.invoke(input="天才AI少女是谁")
```


# 后检索优化

## 1. RAG-Fusion

1. 先生成多个问题，分别获取回答
   - 用不了 `MultiQueryRetriever`，因为它的回答没有标注是哪个问题的
2. 进行rrf计算


## 2. 上下文压缩

将筛选出的文档进行压缩/过滤
- 使用 `ContextualCompressionRetriever` 结合 retriever 使用
- 使用 `DocumentCompressorPipeline` 串联多个压缩器
  - `LLMChainExtractor` 基于 LLM 的压缩器，影响性能
  - `LLMChainFilter` 基于 LLM 的过滤器，不修改文档
  - `EmbeddingsFilter` 基于向量的过滤器，可以控制相似度，不修改文档
  - `EmbeddingsRedundantFilter` 基于向量的过滤器，过滤重复的文档


```python
compressor = LLMChainExtractor.from_llm(llm)
filter = LLMChainFilter.from_llm(llm)

# 查询问题和文档的相似度进行过滤
relevant_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.66)

# EmbeddingsRedundantFilter是文档和文档之间进行过滤的压缩器
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)


# splitter 分块、两个 filter
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

compression_retriever.invoke("deepseek的发展历程")
```