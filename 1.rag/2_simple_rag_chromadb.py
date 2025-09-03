import os
from openai import OpenAI
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter # langchain
from langchain_community.document_loaders import TextLoader

'''
准备：读取 -> 分割 -> 向量化 -> 存储
    读取:   TextLoader/WebBaseLoader... .load()[0].page_content
    分割:   RecursiveCharacterTextSplitter.split_text
    向量化: ai_client.embeddings.create
    存储:   collection = chromadb.Client().get_or_create_collection("db_name")
            collection.add(embeddings, documents, ids) # 三个 List 顺序一一对应

查询：匹配 -> 查询
    # 匹配: 余弦距离（越大越好）/欧式距离 L2（越小越好）
    匹配: collection.query(query_embeddings, n_results=3)
    查询: ai_client.chat.completions.create

'''
class VectorDBHandler:
    def __init__(self, file_name, db_name):
        db = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_db"))
        self.collection = db.get_or_create_collection(db_name)
        self.file_path = os.path.join(os.path.dirname(__file__), file_name)
        self.ai_client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"))
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=30,
            chunk_overlap=5
        )

    def get_embedding(self, text):
        return self.ai_client.embeddings.create(input=text, model='text-embedding-v2')

    def prepare(self):
        if not self.collection.get()['ids']:
            print('[reading file...]')
            # 1. 读取
            article = TextLoader(
                file_path=self.file_path,
                encoding='utf-8'
            ).load()[0].page_content

            # 2. 分割
            processed_sentences = self.splitter.split_text(article)
            print(f'句子数量：{len(processed_sentences)}')
            # for i, sentence in enumerate(processed_sentences):
            #     print(f'【{i+1}】 {sentence}')

            # 3. 获取向量（1536维）
            vectors = [x.embedding for x in self.get_embedding(processed_sentences).data]
            self.collection.add(
                embeddings=vectors,  # 每个文档的向量
                documents=processed_sentences,  # 文档的原文
                ids=[f"id{i}" for i in range(len(processed_sentences))]  # 每个文档的 id
            )
            print(f'向量数量：{len(vectors)}，向量维度：{len(vectors[0])}')

    def ask(self, question):
        # 4. 匹配
        similarities = self.collection.query(
            query_embeddings=self.get_embedding(question).data[0].embedding,
            n_results=3,
        )
        # 输出前3个
        rag = '\n'.join([f'{i+1}. {x}' for i,x in enumerate(similarities['documents'][0])])

        # 5. 查询
        response = self.ai_client.chat.completions.create(
            model="qwen-plus",
            messages=[{ 'role': 'system', 'content': '我能帮你查询说明书'},
                {'role': 'user', 'content': f'帮我查询说明书：我的问题是：{question}，下面是说明书相关的部分内容：{rag}。如果内容不相关，请大胆返回不知道'}],
            # 0.1-0.6之间，回复会更贴切与实际情况，0.6-0.9之间，中间值，<2.0，回复的内容更具有随机性，创造性
            temperature=0
        )
        print(f"问题: {question}")
        print(response.choices[0].message.content)


if __name__ == '__main__':
    file_name = r'1_manual.txt'
    db_name = r'manual'
    vectorHandler = VectorDBHandler(file_name, db_name)
    vectorHandler.prepare()
    vectorHandler.ask("盒子里面有什么")
    # vectorHandler.ask("点击按钮没有反应")
    # vectorHandler.ask("现在几点")
