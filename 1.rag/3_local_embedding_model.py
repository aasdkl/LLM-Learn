# 加载模型
from sentence_transformers import SentenceTransformer

local_llm_path = r'C:\dev\llm\local-model\bge-large-zh-v1.5'

model = SentenceTransformer(local_llm_path)

data = ['你好']
# 默认返回的nump数组
embedding = model.encode(data)
print(embedding.shape)



# 加载huggingface模型
from langchain_huggingface import HuggingFaceEmbeddings

# 生成的嵌入向量将被归一化, 有助于向量比较
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=local_llm_path,
    encode_kwargs=encode_kwargs
)
text = "大模型"
query_result = embeddings.embed_query(text)
print(query_result[:5])
