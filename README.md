# bjtunlp: Beijing Jiaotong University Natural Language Processing

基于预训练语言模型Electra实现的汉语自然语言处理工具。目前支持汉语分词、词性标注、与依存句法分析。
目前主流自然语言处理工具采用Pipeline的方式，即先分词再词性标注最后进行依存句法分析，
存在错误传播问题，而本工具采用了联合学习的方式(非multi tasks)将这三个任务统一在基于图的分析方法框架下。
实验结果表明性能较Pipeline的方式有明显提升。


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
