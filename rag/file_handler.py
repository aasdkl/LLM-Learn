import os
import json
import re

class FileHandler:
    @staticmethod
    def read_file(file_path, from_json=False):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                result = file.read()
                if from_json:
                    result = json.loads(result)
                return result


    @staticmethod
    def write_file(file_path, content):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    @staticmethod
    def demo_splitter(content):
        sentences = re.split(r'\n\n+', content)
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
        return processed_sentences
