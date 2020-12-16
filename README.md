# bjtunlp: Beijing Jiaotong University Natural Language Processing

面向生产环境的多语种自然语言处理工具包，基于最新预训练语言模型Electra，目标是普及落地最前沿的NLP技术。
bjtunlp具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。


 ## 安装

```bash
pip install hanlp
python setup.py install
```

要求Python 3.7以上，支持Windows，可以在CPU上运行，推荐GPU。

## 快速上手

### 中文分词

第一步需要从磁盘或网络加载模型文件。比如，此处加载一个名为 `joint.model` 的模型。

```python
>>> from bjtunlp import BJTUNLP
>>> nlp = BJTUNLP('joint.model')
>>> print(nlp.seg(['中国进出口银行与中国银行加强合作']))
[['中国','进出口','银行','与','中国','银行','加强','合作']]
```
### 中文词性标注
```python
>>> from bjtunlp import BJTUNLP
>>> nlp = BJTUNLP('joint.model')
>>> print(nlp.pos(['中国进出口银行与中国银行加强合作']))
[['中国_NR','进出口_NN','银行_NN','与_CC','中国_NR','银行_NN','加强_VV','合作_NN']]
```

### 中文依存句法分析
输出是 CoNLL-X 格式[^conllx]的句法树。
```python
>>> from bjtunlp import BJTUNLP
>>> nlp = BJTUNLP('joint.model')
>>> print(nlp.dep(['中国进出口银行与中国银行加强合作']))
1	中国	_	NR	_	_	3	NMOD	_	_
2	进出口	_	NN	_	_	3	MMOD	_	_
3	银行	_	NN	_	_	7	SBJ	_	_
4	与	_	CC	_	_	3	CJTN	_	_
5	中国	_	NR	_	_	6	NMOD	_	_
6	银行	_	NN	_	_	4	CJT	_	_
7	加强	_	VV	_	_	0	ROOT	_	_
8	合作	_	NN	_	_	7	COMP	_	_
```
