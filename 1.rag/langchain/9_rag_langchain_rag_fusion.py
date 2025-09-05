

import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.load import dumps, loads

texts = [
    "人工智能在医疗诊断中的应用。",
    "人工智能如何提升供应链效率。",
    "NBA季后赛最新赛况分析。",
    "传统法式烘焙的五大技巧。",
    "红楼梦人物关系图谱分析。",
    "人工智能在金融风险管理中的应用。",
    "人工智能如何影响未来就业市场。",
    "人工智能在制造业的应用。",
    "今天天气怎么样",
    "人工智能伦理：公平性与透明度。"
]

# 本地embedding模型地址
embedding_model_path = r'C:\dev\llm\local-model\bge-large-zh-v1.5'
# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)
# 创建向量数据库对象
vectorstore = Chroma.from_texts(
    texts=texts, embedding=embeddings_model
)

# 创建基础检索器
retriever = vectorstore.as_retriever()

# https://smith.langchain.com/hub/search?q=langchain-ai%2Frag-fusion-query-generation
prompt = hub.pull("langchain-ai/rag-fusion-query-generation")
# print(prompt)

llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)
# 创建多重查询chain
generate_queries = (
        prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

original_query = "人工智能的应用"
queries = generate_queries.invoke({"original_query": original_query})
# print(queries)

print('--------------问题检索到的内容-------------')
print(queries)
# 我们检索到的内容的位置是不一样的, 需要给查询到的数据进行融合重排, 和查询优化很像
for i in queries:
    print(retriever.invoke(i))


# results

def reciprocal_rank_fusion(results: list[list], k=60):
    """互逆排序融合算法，用于合并多个排序文档列表
    Args:
        results: 包含多个排序文档列表的二维列表
        k: 融合公式中的平滑参数（默认60），值越小排名影响越大
    Returns:
        按融合分数降序排列的文档列表，每个元素为(文档对象, 分数)元组
    """

    # 初始化融合分数字典（key=序列化文档，value=累计分数）
    fused_scores = {}

    # 遍历每个检索结果列表（每个查询对应的结果）
    for docs in results:
        # 对当前结果列表中的文档进行遍历（rank从0开始计算）
        for rank, doc in enumerate(docs):
            # 序列化文档对象为字符串（用于唯一标识）
            doc_str = dumps(doc)
            # 初始化文档得分（如果是首次出现）
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 计算并累加RRF分数：1 / (当前排名 + k)
            # 排名越靠前（rank值小）的文档获得的分数越高
            fused_scores[doc_str] += 1 / (rank + k)

    print(fused_scores)
    # 按融合分数降序排序（分数越高排名越前）
    sorted_Data = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_Data)
    reranked_results = [
        (loads(doc), score)  for doc, score in sorted_Data
    ]

    return reranked_results

print('--------------问题融合后的内容-------------')
original_query = "人工智能的应用"
# map()     检测4个结果
chain = generate_queries | retriever.map() | reciprocal_rank_fusion

# 输入结果列表
result_list = chain.invoke({"original_query": original_query})
print(result_list)
# 提取文档内容和对应分数
for i in result_list:
    print(i[0].page_content, i[1])

