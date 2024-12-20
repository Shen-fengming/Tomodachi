from transformers import MarianMTModel, MarianTokenizer

def translate_word_local(word, sentence, target_language="en"):
    """
    使用 MarianMT 本地模型翻译单词。
    参数:
        word (str): 需要翻译的单词
        sentence (str): 单词所在的上下文句子
        target_language (str): 翻译的目标语言，默认为英文 "en"
    返回:
        str: 翻译结果
    """
    # 加载日语到英语的 MarianMT 模型和分词器
    model_name = "Helsinki-NLP/opus-mt-ja-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # 对句子进行翻译
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)

    # 解码翻译结果
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text

# 示例调用
if __name__ == "__main__":
    word = "学生"
    sentence = "私は学生です。"
    translation = translate_word_local(word, sentence)
    print(f"单词 '{word}' 的翻译是: {translation}")