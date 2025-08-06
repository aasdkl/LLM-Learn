import re
import os
from openai import OpenAI
import numpy as np
from numpy.linalg import norm
import json
from numpy import dot

# 余弦距离
def cos_sim(a, b):
    """  余弦距离   越大越近"""
    return dot(a, b) / (norm(a) * norm(b))

# 欧式距离
def l2(a, b):
    """  距离关系 越小越相识  """
    x = np.asarray(a) - np.asarray(b)
    return norm(x)

class FileVectorHandler:
    def __init__(self, file_name, db_name):
        self.file_path = os.path.join(os.path.dirname(__file__), file_name)
        self.db_path = os.path.join(os.path.dirname(__file__), db_name)
        self.ai_client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("DASHSCOPE_BASE_URL"))

    def read_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.article = file.read()

    def read_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as file:
                self.db = json.loads(file.read())
        else:
            self.db = None

    def write_db(self, content):
        with open(self.db_path, 'w', encoding='utf-8') as file:
            file.write(content)

    def split_article(self):
        sentences = re.split(r'\n\n+', self.article)
        sentences = [s.strip() for s in sentences if s]
        processed_sentences = []
        for i in range(len(sentences)):
            current = sentences[i]
            # 获取前一个段落的最后5个字符
            prev_context = ""
            if i > 0:
                prev = sentences[i-1].strip()
                prev_context = prev[-5:] if len(prev) >= 5 else prev
            # 获取后一个段落的前5个字符
            next_context = ""
            if i < len(sentences) - 1:
                next_ = sentences[i+1].strip()
                next_context = next_[:5] if len(next_) >= 5 else next_
            # 组合当前段落及其上下文
            result = f"{prev_context}\n{current}\n{next_context}"
            processed_sentences.append(result)
        self.processed_sentences = processed_sentences

    def get_embedding(self, text):
        return self.ai_client.embeddings.create(input=text, model='text-embedding-v2')

    def prepare(self):
        # 1. 读取
        self.read_file()

        # 2. 分割
        self.split_article()
        print(f'句子数量：{len(self.processed_sentences)}')
        # for i, sentence in enumerate(self.processed_sentences):
        #     print(f'【{i+1}】 {sentence}')

        # 3. 获取向量（1536维）
        self.read_db()
        if not self.db:
            vectors = self.get_embedding(self.processed_sentences)
            self.db = [(self.processed_sentences[i], each.embedding) for i, each in enumerate(vectors.data)]
            self.write_db(json.dumps(self.db, indent=4))
        print(f'向量数量：{len(self.db)}，向量维度：{len(self.db[0][1])}')

    def ask(self, question):
        question_vector = self.get_embedding(question).data[0].embedding
        # 4. 匹配
        similarities = [(cos_sim(question_vector, each[1]), each[0]) for each in self.db]
        # 对元素列表，基于第一项排序
        similarities.sort(key=lambda x: x[0], reverse=True)
        # 输出前3个
        rag = '\n'.join([f'{i+1}. {sentence}' for i, (similarity, sentence) in enumerate(similarities[:3])])

        # 5. 查询
        response = self.ai_client.chat.completions.create(
            model="qwen-plus",
            messages=[{ 'role': 'system', 'content': '我能帮你查询说明书'},
                {'role': 'user', 'content': f'帮我查询说明书：我的问题是：{question}，下面是说明书相关的部分内容：{rag}'}],
            # 0.1-0.6之间，回复会更贴切与实际情况，0.6-0.9之间，中间值，<2.0，回复的内容更具有随机性，创造性
            temperature=0.1
        )
        print(response.choices[0].message.content)

if __name__ == '__main__':
    file_name = r'1_manual.txt'
    db_name = r'1_manual-db.txt'
    vectorHandler = FileVectorHandler(file_name, db_name)
    vectorHandler.prepare()
    # vectorHandler.ask("盒子里面有什么")
    vectorHandler.ask("点击按钮没有反应")
