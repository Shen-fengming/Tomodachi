from flask import Flask, request, jsonify
import json
import spacy

# 初始化 Flask 应用
app = Flask(__name__)

# 加载 spaCy + GiNZA 模型
try:
    nlp = spacy.load("ja_ginza")  # 确保 GiNZA 模型已安装
except OSError:
    raise RuntimeError("GiNZA 模型未安装，请运行 'python -m spacy download ja_ginza' 安装模型")

def analyze_text_logic(text):
    """核心处理逻辑，独立为函数"""
    doc = nlp(text)
    return [
        {
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "is_oov": token.is_oov,
            "is_n5": search_word_in_json(token.lemma_, json_file)
        }
        for token in doc
    ]

json_file = "python_translator_server/database/jlpt/n5.json"

def search_word_in_json(word_to_search, json_file):
    # 加载 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 查询单词
    for entry in data:
        if entry.get("word") == word_to_search:
            return {
                "found": True,
                "details": entry
            }

    return {"found": False}


@app.route('/analyze', methods=['POST'])
def analyze_text():
    # 从请求中获取日语文本
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input. 'text' is required."}), 400

    text = data['text']
    doc = nlp(text)

    # 处理分词和词性标注
    tokens = [
        {
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "is_oov": token.is_oov,  # 标记是否为未知词汇
            "is_n5": search_word_in_json(token.lemma_, json_file)
        }
        for token in doc
    ]

    return jsonify({"tokens": tokens}), 200
    # return app.response_class(
    #     response=jsonify({"tokens": tokens}).get_data(as_text=True),
    #     mimetype="application/json",
    #     content_type="application/json; charset=utf-8"
    # )


# 启动 Flask 服务
if __name__ == '__main__':
    app.run(debug=True)