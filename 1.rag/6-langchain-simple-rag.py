import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

'''
准备：读取 -> 分割 -> 向量化 -> 存储
    （不变）
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
        self.db_name = db_name
        self.file_path = os.path.join(os.path.dirname(__file__), file_name)
        self.db_dir = os.path.dirname(__file__)
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=30,
            chunk_overlap=5
        )
        self.embedding_func = DashScopeEmbeddings(dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))
        self.rag_chain = self._build_rag_chain()

    def _build_rag_chain(self):
        llm = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            model='qwen-plus',
            temperature=0
        )
        template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("我能帮你查询说明书"),
            HumanMessagePromptTemplate.from_template('帮我查询说明书：我的问题是：{question}，下面是说明书相关的部分内容：{rag}。如果内容不相关，请大胆返回不知道')
        ])
        return template | llm


    def get_embedding(self, text):
        return self.ai_client.embeddings.create(input=text, model='text-embedding-v2')

    def prepare(self):
        db = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embedding_func,
            collection_name=self.db_name
        )

        if not db.get()['ids']:
            print('[reading file...]')
            # 1. 读取
            article = TextLoader(
                file_path=self.file_path,
                encoding='utf-8'
            ).load()[0].page_content

            # 2. 分割
            paragraphs = self.splitter.create_documents([article])
            print(f'句子数量：{len(paragraphs)}')
            # for i, sentence in enumerate(processed_sentences):
            #     print(f'【{i+1}】 {sentence}')

            # 3. 获取向量（1536维）
            db = Chroma.from_documents(paragraphs, self.embedding_func,
                                       persist_directory=self.db_dir, collection_name=self.db_name)
            db.persist()
            # print(f'向量数量：{len(vectors)}，向量维度：{len(vectors[0])}')

        self.retriever = db.as_retriever(search_kwargs={"k": 3})

    def ask(self, question):
        # 4. 匹配
        similarities = self.retriever.get_relevant_documents(question)
        # 输出前3个
        rag = '\n'.join([f'{i+1}. {x.page_content}' for i,x in enumerate(similarities)])

        # 5. 查询
        response = self.rag_chain.invoke({"question": question, "rag": rag})
        print(response.content)


if __name__ == '__main__':
    file_name = r'1_manual.txt'
    db_name = r'manual-langchain'
    vectorHandler = VectorDBHandler(file_name, db_name)
    vectorHandler.prepare()
    vectorHandler.ask("盒子里面有什么")
    # vectorHandler.ask("点击按钮没有反应")
    # vectorHandler.ask("现在几点")
