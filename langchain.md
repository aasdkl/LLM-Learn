高级 RAG

预检索
1. 优化索引
   1. 摘要索引 MultiVectorRetriever
   2. 父子索引 ParentDocumentRetriever
   3. 假设性问题 - 不常用，因为问题不太确定
   4. 元数据索引 - SelfQueryRetriever 类似摘要，但是存储的是字段（一般英文的元数据更准确
   5. 混合检索 - EnsembleRetriever 传统搜索算法（Best Matching 25, BM25 BM25Retriever）与向量相似性检索相结合
2. 扩展索引
   1. 将用户提问扩展 MultiQueryRetriever

后检索
1. RAG-Fusion 融合查出来的文档，在Multi Query的基础上，对其检索结果进行重新排序(即reranking)后输出Top K个最相关文档，最后将这top k个文档喂给LLM并生成最终的答案 
2. 上下文压缩 - 将筛选出的文档进行压缩/过滤（使用 ContextualCompressionRetriever 结合使用）
   - LLMChainExtractor 基于 LLM 的压缩器，影响性能
   - LLMChainFilter 基于 LLM 的过滤器，不修改文档
   - EmbeddingsFilter 基于向量的过滤器，可以控制相似度，不修改文档

DocumentCompressorPipeline 可以使用多个压缩器