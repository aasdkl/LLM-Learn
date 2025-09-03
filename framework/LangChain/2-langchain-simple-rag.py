import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma  # 使用更新后的包
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatTongyi  # 专为DashScope设计的聊天模型
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

'''
准备：读取 -> 分割 -> 向量化 -> 存储
    读取:   TextLoader/WebBaseLoader... .load()[0].page_content
    分割:   RecursiveCharacterTextSplitter.split_text
    向量化、存储:   Chroma.from_documents

查询：匹配 -> 查询
    匹配:   通过获取 retriever，调用 .invoke()
    查询:   prompt|llm 管道进行 .invoke()，可以和匹配合并使用

'''
class VectorDBHandler:
    def __init__(self, file_name, db_name):
        self.db_name = db_name
        self.file_path = os.path.join(os.path.dirname(__file__), file_name)
        self.db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")  # 单独目录存储
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=30,
            chunk_overlap=5
        )
        self.embedding_func = DashScopeEmbeddings(
            dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="text-embedding-v2")

    def _build_rag_chain(self):
        llm = ChatOpenAI(  # ChatTongyi
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            model='qwen-plus',
            temperature=0
        )

        template = """帮我根据提供的上下文回答问题。
        上下文内容：
        {rag}

        问题：{question}
        如果上下文与问题无关，请回答'我不知道'。"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("我能帮你查询说明书"),
            HumanMessagePromptTemplate.from_template(template)
        ])

        # 这种写法将自动获取 rag 的内容，如果需要得到参考文档，则不使用管道第一项
        return (
            {"rag": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

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
            # print(f'向量数量：{len(vectors)}，向量维度：{len(vectors[0])}')

        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.rag_chain = self._build_rag_chain()

    def ask(self, question):
        # 4. 直接查询，底层进行匹配
        response = self.rag_chain.invoke(question)
        print(f"问题: {question}")
        print(response.content)
        print(self.retriever.invoke(question)) # 参考文档


if __name__ == '__main__':
    file_name = r'1_manual.txt'
    db_name = r'manual-langchain'
    vectorHandler = VectorDBHandler(file_name, db_name)
    vectorHandler.prepare()
    vectorHandler.ask("盒子里面有什么")
    # vectorHandler.ask("点击按钮没有反应")
    # vectorHandler.ask("现在几点")
