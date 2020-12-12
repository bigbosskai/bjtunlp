import torch


class BJTUNLP(object):
    def __init__(self, path, device=None, **kwargs):
        if device is not None:
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.__model = torch.load(path, map_location=self.device)

    def __predict(self, text):
        """
        ['中国进出口银行与中国银行加强合作','新华社北京电']
        :param text:
        :return:
        """
        text_chars = []
        for sent in text:
            text_chars.append(list(sent))
        res = self.__model.predict_text(text_chars)
        return res

    def __convert_chars(self, text):
        text_chars = []
        for sent in text:
            text_chars.append(list(sent))
        return text_chars

    def seg(self, text):
        text_chars = self.__convert_chars(text)
        res = self.__predict(text_chars)
        return res[0]

    def pos(self, text):
        text_chars = self.__convert_chars(text)
        res = self.__predict(text_chars)
        ret = []
        for tokens, pos_seq in zip(res[0], res[1]):
            sent = []
            for tok, pos in zip(tokens, pos_seq):
                sent.append(tok + '_' + pos)
            ret.append(sent)
        return ret

    def dep(self, text):
        """
        返回CoNLL格式的依存句法分析样式
        1	中国	_	NR	_	_	3	NMOD	_	_
        2	进出口	_	NN	_	_	3	NMOD	_	_
        3	银行	_	NN	_	_	7	SBJ	_	_
        4	与	_	CC	_	_	3	CJTN	_	_
        5	中国	_	NR	_	_	6	NMOD	_	_
        6	银行	_	NN	_	_	4	OBJ	_	_
        7	加强	_	VV	_	_	0	ROOT	_	_
        8	合作	_	NN	_	_	7	OBJ	_	_
        :param text:
        :return: CoNLL format
        """
        text_chars = self.__convert_chars(text)

        res = self.__predict(text_chars)
        ret = []
        for tokens, pos_seq, heads, deps in zip(res[0], res[1], res[2], res[3]):
            sent = []
            for i, (tok, pos, h, dep) in enumerate(zip(tokens, pos_seq, heads, deps)):
                sent.append(f"{i + 1}\t{tok}\t_\t{pos}\t_\t_\t{h}\t{dep}\t_\t_".format(i, tok, pos, h, dep))
            s = "\n".join(sent)
            ret.append(s)
        return '\n\n'.join(ret)


if __name__ == '__main__':
    bjtunlp = BJTUNLP('../epoch.pkl')
    print(bjtunlp.seg(['中国进出口银行与中国银行加强合作', '新华社北京电', '我喜欢你', '我爱唐续文']))
    print(bjtunlp.pos(['中国进出口银行与中国银行加强合作', '新华社北京电', '我喜欢你', '我爱唐续文']))
    print(bjtunlp.dep(['中国进出口银行与中国银行加强合作', '新华社北京电', '我喜欢你', '我爱唐续文', '研究生活好']))
