from typing import *
import torch.nn as nn
import torch
from transformers import ElectraTokenizer, ElectraModel


class ElectraEmbedding(nn.Module):
    def __init__(self, vocab: str, path: str):
        """
        完成用户创建的词输入[batch_size, seq_len] 经过Electra模型输出为[batch_size, seq_len, embedding_size]为上下文表示
        :param vocab: 用户自定义的词表，其中必须包含idx到word的字典映射关系，以及padding_idx
        :param path: 从huggingface下载的Electra语言模型
        """
        super(ElectraEmbedding, self).__init__()
        self.vocab = vocab
        self.tokenizer = ElectraTokenizer.from_pretrained(path)
        self.embed = ElectraModel.from_pretrained(path, return_dict=True)
        self.embed_size = self.embed.config.embedding_size

    def predict(self, input_tokens: List[List[str]]):
        """
        这里的input_ids不是BERT的index表示，应为仿照fastNLP的做法，其内部作了两个词表之间的映射关系
        [['上', '海', '浦', '东', '开', '发', '与', '法', '制', '建', '设', '同', '步'],
        ['青', '岛', '未', '发', '现', '新', '增', '阳', '性']]
        :return: the last layer of Electra model, the shape is [batch_size, input_tokens_max_len, hidden_size]
        """

        input_tokens_max_len = max(map(len, input_tokens))
        orig_to_tok_map = []
        bert_tokens = []
        for tokens in input_tokens:
            orig_map = []
            bert_token = ["[CLS]"]
            for token in tokens:
                orig_map.append(len(bert_token))
                bert_token.extend(self.tokenizer.tokenize(token))
            bert_token.append("[SEP]")
            orig_to_tok_map.append(orig_map)
            bert_tokens.append(bert_token)

        max_len = max(map(len, bert_tokens))
        padded_tokens = []
        for tokens in bert_tokens:
            padded_tokens.append(tokens + ['[PAD]'] * (max_len - len(tokens)))

        inputs = self.tokenizer(padded_tokens, return_tensors="pt", is_pretokenized=True, padding=False,
                                add_special_tokens=False)
        # device = self.embed.device
        inputs = {k: v.to(self.embed.device) for k, v in inputs.items()}
        outputs = self.embed(**inputs)
        last_hidden_states = outputs.last_hidden_state
        res = []
        for idx, position in enumerate(orig_to_tok_map):
            res.append(last_hidden_states[idx][position + [-1] * (input_tokens_max_len - len(position))])
        ret_hidden_state = torch.stack(res, dim=0)  # 恢复成[batch_size, input_tokens_max_len, hidden_size]
        return ret_hidden_state



    def forward(self, input_tokens: List[List[str]]):
        """
        这里的input_ids不是BERT的index表示，应为仿照fastNLP的做法，其内部作了两个词表之间的映射关系
        [['上', '海', '浦', '东', '开', '发', '与', '法', '制', '建', '设', '同', '步'],
        ['青', '岛', '未', '发', '现', '新', '增', '阳', '性']]
        :return: the last layer of Electra model, the shape is [batch_size, input_tokens_max_len, hidden_size]
        """
        shape = input_tokens.size()
        input_tokens = input_tokens.tolist()
        raw_tokens = []
        for tokens in input_tokens:
            raw_chars = []
            for tok_idx in tokens:
                if tok_idx != self.vocab.padding_idx:
                    if tok_idx == self.vocab.unknown_idx:
                        raw_chars.append('[UNK]')
                        continue
                    raw_chars.append(self.vocab.idx2word[tok_idx])
            raw_tokens.append(raw_chars)

        input_tokens_max_len = max(map(len, raw_tokens))
        orig_to_tok_map = []
        bert_tokens = []
        for tokens in raw_tokens:
            orig_map = []
            bert_token = ["[CLS]"]
            for token in tokens:
                orig_map.append(len(bert_token))
                bert_token.extend(self.tokenizer.tokenize(token))
            bert_token.append("[SEP]")
            orig_to_tok_map.append(orig_map)
            bert_tokens.append(bert_token)

        max_len = max(map(len, bert_tokens))
        padded_tokens = []
        for tokens in bert_tokens:
            padded_tokens.append(tokens + ['[PAD]'] * (max_len - len(tokens)))

        inputs = self.tokenizer(padded_tokens, return_tensors="pt", is_pretokenized=True, padding=False,
                                add_special_tokens=False)
        # device = self.embed.device
        inputs = {k: v.to(self.embed.device) for k, v in inputs.items()}
        outputs = self.embed(**inputs)
        last_hidden_states = outputs.last_hidden_state
        res = []
        for idx, position in enumerate(orig_to_tok_map):
            res.append(last_hidden_states[idx][position + [-1] * (input_tokens_max_len - len(position))])
        ret_hidden_state = torch.stack(res, dim=0)  # 恢复成[batch_size, input_tokens_max_len, hidden_size]
        assert ret_hidden_state.shape[:2] == shape
        return ret_hidden_state

# embed = ElectraEmbedding(None, r'H:\预训练语言模型\哈工大20G语料-Electra\base\discriminator')
# #
# hidden_state = embed.predict([['上', '海', '浦', '东', '开', '发', '与', '法', '制', '建', '设', '同', '脦'],
#                       ['青', '岛', '未', '发', '现', '新', '增', '阳', '性']])
# print(hidden_state.size())

# for name, param in embed.named_parameters():
#         if param.requires_grad:
#             print(name,':',param.size())
