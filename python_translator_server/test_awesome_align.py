from transformers import AutoTokenizer, AutoModel
import torch

class AwesomeAligner:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        # 初始化分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def align(self, source_sentence, target_sentence):
        """
        对齐源句子和目标句子。
        参数:
            source_sentence (str): 源句子（例如：日语）。
            target_sentence (str): 目标句子（例如：英语）。
        返回:
            list: 对齐结果 [(源词索引, 目标词索引), ...]
        """
        # 对源句和目标句进行分词
        inputs = self.tokenizer(
            source_sentence,
            text_pair=target_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        print(input)

        # 获取对齐矩阵
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions[-1]  # 使用最后一层的注意力

        # 计算对齐
        attention_matrix = attentions.mean(dim=1)[0].detach().cpu().numpy()  # 平均头部的注意力权重
        alignments = self._extract_alignments(inputs, attention_matrix)
        return alignments

    def _extract_alignments(self, inputs, attention_matrix, threshold=0.2):
        """
        从注意力矩阵中提取对齐信息。
        """
        # 分离源语言和目标语言的 token
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        sep_index = input_tokens.index("[SEP]")  # 找到分隔符
        source_tokens = input_tokens[1:sep_index]  # 源语言 tokens
        target_tokens = input_tokens[sep_index + 1:-1]  # 目标语言 tokens

        # 提取对齐信息
        alignments = []
        for i, src_token in enumerate(source_tokens):
            for j, tgt_token in enumerate(target_tokens):
                if attention_matrix[i, j] > threshold:
                    alignments.append((i, j))
        return alignments


# 示例调用
if __name__ == "__main__":
    aligner = AwesomeAligner()
    source_sentence = "私は学生です。"
    target_sentence = "I am a student."
    alignments = aligner.align(source_sentence, target_sentence)
    print(f"对齐结果: {alignments}")