
# pip install ragas

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI


# 创建 embedding
embedding_model_path = r'C:\dev\llm\local-model\bge-large-zh-v1.5'
bge_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
# 获取数据
urls = "https://baike.baidu.com/item/%E6%81%90%E9%BE%99/139019"
loader = WebBaseLoader(urls, header_template={
    'Cookie': 'zhishiTopicRequestTime=1751201782079; PSTM=1751188919; BAIDUID=8CA6687616903FB8DD08FDF59F6CDB19:FG=1; BAIDUID_BFESS=8CA6687616903FB8DD08FDF59F6CDB19:FG=1; BA_HECTOR=ag8k808l8ha080850k2ha40hah0l201k621dp25; BIDUPSID=BB3C8281FAB5D9BF613466F32412F0C4; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; ZFY=5DIcVpIF22itPpGW:BVZTxbnb28NTOgUhWwfF2ZU9I:BY:C; baikeVisitId=be8175c8-fe31-4b1f-b7a7-07e72d81316d; ab_sr=1.0.1_YTQxY2Q4ZDgxNjUwYmE3NThlZTMxNGU0ODkzNzM1MGVlNjBjOWIwM2ZlOWQ1NTJkYmYzNmQwN2M3Mjg4OTNmMmIxYzk1NWM3NzNkZjUzZjgyYjg2ODdkZTRlN2NlMzVkZmE3OWIyY2E3MzA3Y2FjNGFkMDNhN2E1NjRhNGU2ZGM5MmZkZjJlYWRiMDFiNGQ0MGZlMTIxNjE2OTJkNTBiYw==; H_PS_PSSID=60271_62327_63141_63327_63582_63576_63636_63646_63655_63676_63724_63727_63719',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
})
docs = loader.load()

# 创建主文档分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# 创建子文档分割器   一个父文档里面会得到几个子文档   200    1000    0-400    200-600   400 - 800  600-100
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# 创建向量数据库对象
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=bge_embeddings
)

# 创建内存存储对象
store = InMemoryStore()
# 创建父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,   # 向量数据库
    docstore=store,    # 存父文档的内存
    child_splitter=child_splitter,   # 子分割器
    parent_splitter=parent_splitter,
    #     verbose=True,
    search_kwargs={"k": 2}
)
# 添加文档集
retriever.add_documents(docs)

chat = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)

# 创建prompt模板
template = """你是负责回答问题的助手。使用以下检索到的上下文片段来回答问题。
如果你不知道答案，就说你不知道。最多用两句话，回答要简明扼要。
Question: {question} 
Context: {context} 
Answer:
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)
# 创建chain
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | chat | StrOutputParser()

chain.invoke({"question":"恐龙是什么?"})

from datasets import Dataset

# 问题
questions = ["恐龙是怎么被命名的？",
             "恐龙怎么分类的？",
             "体型最大的是哪种恐龙?",
             "体型最长的是哪种恐龙？它在哪里被发现？",
             "恐龙采样什么样的方式繁殖？",
             "恐龙是冷血动物吗？",
             "陨石撞击是导致恐龙灭绝的原因吗？",
             "恐龙是在什么时候灭绝的？",
             "鳄鱼是恐龙的近亲吗？",
             "鸡是恐龙的近亲吗？",
             "恐龙在英语中叫什么？"
             ]
# 真实答案
ground_truths = [
    "1841年，英国科学家理查德·欧文在研究几块样子像蜥蜴骨头化石时，认为它们是某种史前动物留下来的，并命名为恐龙，意思是“恐怖的蜥蜴”。",
    "恐龙可分为鸟类和非鸟恐龙。",
    "恐龙整体而言的体型很大。以恐龙作为标准来看，蜥脚下目是其中的巨无霸。",
    "最长的恐龙是27米长的梁龙，是在1907年发现于美国怀俄明州。",
    "恐龙采样产卵、孵蛋的方式繁殖。",
    "恐龙是介于冷血和温血之间的动物",
    "科学家最新研究显示，0.65亿年前小行星碰撞地球时间或早或晚都可能不会导致恐龙灭绝，真实灭绝原因是当时恐龙处于较脆弱的生态系统中，环境剧变易导致灭绝。",
    "恐龙灭绝的时间是在距今约6500万年前，地质年代为中生代白垩纪末或新生代第三纪初。",
    "鳄鱼是另一群恐龙的现代近亲，但两者关系较非鸟恐龙与鸟类远。",
    "鸡是恐龙的后代，因为鸡是鸟类，而鸟类演化自恐龙的一个分支——手盗龙类。",
    "1842年，英国古生物学家理查德·欧文创建了“dinosaur”这一名词。英文的dinosaur来自希腊文deinos（恐怖的）Saurosc（蜥蜴或爬行动物）。对当时的欧文来说，这“恐怖的蜥蜴”或“恐怖的爬行动物”是指大的灭绝的爬行动物（实则不是）"
]
# 模型回答
answers = []
# 文档内容
contexts = []

# 把检索到的内容和回答的问题进行存储
for query in questions:
    answers.append(chain.invoke({"question": query}))
    contexts.append([docs.page_content for docs in retriever.invoke(query)])
print("question", questions)
print("answer", answers)
print("contexts", contexts)
print("ground_truth", ground_truths)
# 转换成字典
data_samples = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

# 字典转换为Dataset对象，便于高效处理数据并适配模型训练、评估等任务。
dataset = Dataset.from_dict(data_samples)
print(dataset)


from ragas import evaluate
from ragas.metrics import faithfulness,answer_relevancy,context_recall,context_precision

# 进行评估
result = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=chat,
    embeddings=bge_embeddings
)
df = result.to_pandas()
print(df)
df.to_csv('ragas_reval.csv', index=True)

# | **精确度** (Context Precision) 	    | 检索结果与问题的相关比例  | 几乎所有值都接近1
# | **召回率** (Context Recall) 	    | 关键信息是否被完整检索    | 有多处得分为0，未能全面、充分地利用上下文
# | **忠诚度** (Faithfulness)   	    | 生成答案与检索内容一致性  | 大部分为1，非1的和上下文有细微出入
# | **答案相关性** (Answer Relevance)   | 回答与问题匹配度          | 普遍不是1，过于简略
#                             |  Context   | Context |          | Answer    |
# | 编号 | 问题概要            |  Precision | Recall  | Faithful | Relevancy | 主要原因分析                                                                               |
# | :--- | :----------------- | :--------- | :-------| :------- | :---------| :---------------------------------------------------------------------------------------- |
# | 0    | 恐龙命名            | 0.999      | 1.0     | 1.0      | 0.751     | **Answer Relevancy偏低**: 回答虽然正确，但过于简略，未充分展开命名细节。                      |
# | 1    | 恐龙分类            | 0.999      | **0.0** | 1.0      | 0.604     | **Context Recall为0**: 回答未包含检索上下文中提供的详细分类信息（如兽脚亚目等）。               |
# | 2    | 最大恐龙            | 0.999      | 1.0     | **0.8**  | 0.866     | **Faithfulness为0.8**: 回答未提及检索上下文中“易碎双腔龙可能更大”这一信息，存在遗漏。          |
# | 3    | 最长恐龙及发现地     | 0.999      | 1.0     | 1.0      | 0.842     | **Answer Relevancy偏低**: 回答正确但非常简短，未补充任何背景信息。                           |
# | 4    | 恐龙繁殖方式         | 0.999      | 1.0     | 1.0      | 0.695     | **Answer Relevancy偏低**: 回答准确但过于简洁，未纳入检索上下文中的细节（如蛋的尺寸、排列方式）。|
# | 5    | 恐龙是否冷血动物     | 0.999      | 1.0     | 1.0      | 0.884     | **Answer Relevancy偏高但非1**: 回答很好地概括了结论，但可能未完全涵盖所有支持证据或细微差别。  |
# | 6    | 陨石撞击与恐龙灭绝   | 0.999      | **0.0** | 1.0      | 0.953     | **Context Recall为0**: 回答完全未提及检索上下文中提供的其他灭绝假说（如火山爆发、气候变迁说）。  |
# | 7    | 恐龙灭绝时间        | 0.999      | 1.0     | 1.0      | 0.809     | **Answer Relevancy偏低**: 回答正确但过于直接，未能联系灭绝事件的影响或争议。                   |
# | 8    | 鳄鱼是否是恐龙近亲   | 0.999      | 1.0    | 1.0       | 0.999     | 各项指标接近满分。回答准确且相关。                                                           |
# | 9    | 鸡是否是恐龙近亲    | 0.999      | 1.0     | 1.0       | -         | Answer Relevancy为空值，无法分析。其他指标满分。                                             |
# | 10   | 恐龙的英文名        | 0.999      | 0.666   | 1.0      | 0.893     | **Context Recall为0.666**: 回答正确，但未能全部利用或体现检索上下文中关于命名的所有相关信息。    |