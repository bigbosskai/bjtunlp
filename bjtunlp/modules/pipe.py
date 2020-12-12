from fastNLP.io.data_bundle import DataBundle
from fastNLP.io import ConllLoader, Pipe
import numpy as np

from itertools import chain
from fastNLP import DataSet, Vocabulary
from functools import partial
from fastNLP.io.utils import check_loader_paths
from .conll import CoNLL
from bjtunlp.modules.alg import numericalize_sibs


class CTBxJointPipe(Pipe):
    """
    文件夹下应该具有以下的文件结构
        -train.conll
        -dev.conll
        -test.conll
    每个文件中的内容如下（空格隔开不同的句子, 共有）
        1	中国	_	NR	NR	_	3	nn	_	_
        2	进出口	_	NN	NN	_	3	nn	_	_
        3	银行	_	NN	NN	_	6	conj	_	_
        4	与	_	CC	CC	_	6	cc	_	_
        5	中国	_	NR	NR	_	6	nn	_	_
        6	银行	_	NN	NN	_	7	nsubj	_	_
        7	加强	_	VV	VV	_	0	root	_	_
        8	合作	_	NN	NN	_	7	dobj	_	_
        ...
    """

    def __init__(self):
        self._loader = ConllLoader(headers=['words', 'pos_tags', 'heads', 'labels'], indexes=[1, 3, 6, 7])

    def load(self, path: str):
        """
        给定一个文件路径，将数据读取为DataSet格式。DataSet中包含以下的内容
        words: list[str]
        pos_tags: list[str]
        heads: list[int]
        labels: list[str]

        :param path:
        :return: dataset格式的数据集
        """
        dataset = self._loader._load(path)
        dataset.heads.int()
        return dataset

    def process_from_file(self, paths):
        """
        
        :param paths: 
        :return:
            Dataset包含以下的field
                chars:
                bigrams:
                trigrams:
                pre_chars:
                pre_bigrams:
                pre_trigrams:
                seg_targets:
                seg_masks:
                seq_lens:
                char_labels:
                char_heads:
                gold_word_pairs:
                seg_targets:
                seg_masks:
                char_labels:
                char_heads:
                pun_masks:
                gold_label_word_pairs:
        """
        paths = check_loader_paths(paths)
        data = DataBundle()

        for name, path in paths.items():
            dataset = self.load(path)
            data.datasets[name] = dataset

        char_labels_vocab = Vocabulary(padding=None, unknown=None)
        pos_labels_vocab = Vocabulary(padding=None, unknown=None)

        def process(dataset):
            dataset.apply(add_word_lst, new_field_name='word_lst')
            dataset.apply(lambda x: list(chain(*x['word_lst'])), new_field_name='chars')
            dataset.apply(add_bigram, field_name='chars', new_field_name='bigrams')
            dataset.apply(add_trigram, field_name='chars', new_field_name='trigrams')
            dataset.apply(add_char_heads, new_field_name='char_heads')
            dataset.apply(add_pos, new_field_name='char_pos')
            dataset.apply(add_sibs, new_field_name='sibs')
            dataset.apply(add_char_labels, new_field_name='char_labels')
            dataset.apply(add_segs, new_field_name='seg_targets')
            dataset.apply(add_mask, new_field_name='seg_masks')
            dataset.add_seq_len('chars', new_field_name='seq_lens')
            dataset.apply(add_pun_masks, new_field_name='pun_masks')
            return dataset

        def index_dataset(dataset, char_label_vocab, pos_labels_vocab):
            char_label_vocab.index_dataset(dataset, field_name='char_labels')
            pos_labels_vocab.index_dataset(dataset, field_name='char_pos')
            new_dataset = add_root(dataset)
            new_dataset.apply(add_word_pairs, new_field_name='gold_word_pairs', ignore_type=True)
            global add_label_word_pairs
            add_label_word_pairs = partial(add_label_word_pairs, label_vocab=char_label_vocab)
            new_dataset.apply(add_label_word_pairs, new_field_name='gold_label_word_pairs', ignore_type=True)
            new_dataset.set_pad_val('char_labels', -1)
            new_dataset.set_pad_val('char_heads', -1)
            new_dataset.set_pad_val('char_pos', -1)
            new_dataset.set_pad_val('sibs', 0)
            return new_dataset

        for name in ['train', 'dev', 'test']:
            dataset = data.datasets[name]
            print('Loading %s' % name)
            dataset = process(dataset)
            if name == 'train':
                dataset.drop(delete_no_projective,
                             inplace=True)  # delete not projective tree for crf loss recommended by Yu Zhang
            data.datasets[name] = dataset

        special_root = None  # 遍历获得头结点是谁
        for h, dep in zip(data.datasets['train'][0]['heads'], data.datasets['train'][0]['labels']):
            if 0 == h:
                special_root = dep
                break

        # 统计需要预测的标签，形成标签表
        char_labels_vocab.from_dataset(*data.datasets.values(), field_name='char_labels')
        pos_labels_vocab.from_dataset(*data.datasets.values(), field_name='char_pos')

        for name in ['train', 'dev', 'test']:
            dataset = data.datasets[name]
            dataset = index_dataset(dataset, char_labels_vocab, pos_labels_vocab)
            data.datasets[name] = dataset

        data.vocabs['char_labels'] = char_labels_vocab
        data.vocabs['char_pos'] = pos_labels_vocab

        char_vocab = Vocabulary(min_freq=2).from_dataset(data.datasets['train'], field_name='chars',
                                                         no_create_entry_dataset=[data.get_dataset('dev'),
                                                                                  data.get_dataset('test')])
        bigram_vocab = Vocabulary(min_freq=3).from_dataset(data.datasets['train'], field_name='bigrams',
                                                           no_create_entry_dataset=[data.get_dataset('dev'),
                                                                                    data.get_dataset('test')])
        trigram_vocab = Vocabulary(min_freq=5).from_dataset(data.datasets['train'], field_name='trigrams',
                                                            no_create_entry_dataset=[data.get_dataset('dev'),
                                                                                     data.get_dataset('test')])

        for name in ['chars', 'bigrams', 'trigrams']:
            vocab = Vocabulary().from_dataset(field_name=name, no_create_entry_dataset=list(data.datasets.values()))
            vocab.index_dataset(*data.datasets.values(), field_name=name, new_field_name='pre_' + name)
            data.vocabs['pre_{}'.format(name)] = vocab

        for name, vocab in zip(['chars', 'bigrams', 'trigrams'],
                               [char_vocab, bigram_vocab, trigram_vocab]):
            vocab.index_dataset(*data.datasets.values(), field_name=name, new_field_name=name)
            data.vocabs[name] = vocab

        for name, dataset in data.datasets.items():
            dataset.set_input('chars', 'bigrams', 'trigrams', 'seq_lens', 'char_labels', 'char_heads', 'pre_chars',
                              'pre_bigrams', 'pre_trigrams', 'char_pos', 'sibs')
            dataset.set_target('gold_word_pairs', 'seq_lens', 'seg_targets', 'seg_masks', 'char_labels',
                               'char_heads',
                               'pun_masks', 'gold_label_word_pairs', 'char_pos', 'sibs')

        return data, special_root


def add_label_word_pairs(instance, label_vocab):
    # List[List[((head_start, head_end], (dep_start, dep_end]), ...]]
    word_end_indexes = np.array(list(map(len, instance['word_lst'])))
    word_end_indexes = np.cumsum(word_end_indexes).tolist()
    word_end_indexes.insert(0, 0)
    word_pairs = []
    labels = instance['labels']
    pos_tags = instance['pos_tags']
    for idx, head in enumerate(instance['heads']):
        if pos_tags[idx] == 'PU':  # 如果是标点符号，就不记录
            continue
        # label = label_vocab.to_index(labels[idx])  # 他这里的依存标签采用了下标
        label = labels[idx]
        if head == 0:
            word_pairs.append((('root', label, (word_end_indexes[idx], word_end_indexes[idx + 1]))))
        else:
            word_pairs.append(((word_end_indexes[head - 1], word_end_indexes[head]), label,
                               (word_end_indexes[idx], word_end_indexes[idx + 1])))
    return word_pairs


def delete_no_projective(instance):
    heads = instance['heads']
    return not CoNLL.isprojective([0] + heads)


def add_word_pairs(instance):
    # List[List[((head_start, head_end], (dep_start, dep_end]), ...]]
    word_end_indexes = np.array(list(map(len, instance['word_lst'])))
    word_end_indexes = np.cumsum(word_end_indexes).tolist()
    word_end_indexes.insert(0, 0)
    word_pairs = []
    pos_tags = instance['pos_tags']
    for idx, head in enumerate(instance['heads']):
        if pos_tags[idx] == 'PU':  # 如果是标点符号，就不记录
            continue
        if head == 0:
            word_pairs.append((('root', (word_end_indexes[idx], word_end_indexes[idx + 1]))))
        else:
            word_pairs.append(((word_end_indexes[head - 1], word_end_indexes[head]),
                               (word_end_indexes[idx], word_end_indexes[idx + 1])))
    return word_pairs


def add_pos(instance):
    """
    author: bosskai 构造字符级的词性预测任务
    :param instance:
    :return:
    """
    poss = []
    words = instance['word_lst']
    pos_tags = instance['pos_tags']
    assert len(words) == len(pos_tags)
    for ws, p in zip(words, pos_tags):
        poss.extend([p] * len(ws))
    return poss


def add_root(dataset):
    new_dataset = DataSet()
    for sample in dataset:
        chars = ['$'] + sample['chars']
        bigrams = ['$'] + sample['bigrams']
        trigrams = ['$'] + sample['trigrams']
        seq_lens = sample['seq_lens'] + 1
        char_labels = [0] + sample['char_labels']
        char_heads = [0] + sample['char_heads']
        char_pos = [0] + sample['char_pos']
        sibs = [0] + sample['sibs']
        sample['chars'] = chars
        sample['bigrams'] = bigrams
        sample['trigrams'] = trigrams
        sample['seq_lens'] = seq_lens
        sample['char_labels'] = char_labels
        sample['char_heads'] = char_heads
        sample['char_pos'] = char_pos
        sample['sibs'] = sibs
        new_dataset.append(sample)
    return new_dataset


def add_sibs(instance):
    char_heads = instance['char_heads']
    sibs = numericalize_sibs(char_heads)
    return sibs


def add_pun_masks(instance):
    tags = instance['pos_tags']
    pun_masks = []
    for word, tag in zip(instance['words'], tags):
        if tag == 'PU':
            pun_masks.extend([1] * len(word))
        else:
            pun_masks.extend([0] * len(word))
    return pun_masks


def add_word_lst(instance):
    words = instance['words']
    word_lst = [list(word) for word in words]
    return word_lst


def add_bigram(instance):
    chars = instance['chars']
    length = len(chars)
    chars = chars + ['<eos>']
    bigrams = []
    for i in range(length):
        bigrams.append(''.join(chars[i:i + 2]))
    return bigrams


def add_trigram(instance):
    chars = instance['chars']
    length = len(chars)
    chars = chars + ['<eos>'] * 2
    trigrams = []
    for i in range(length):
        trigrams.append(''.join(chars[i:i + 3]))
    return trigrams


def add_char_heads(instance):
    words = instance['word_lst']
    heads = instance['heads']
    char_heads = []
    char_index = 1  # 因此存在root节点所以需要从1开始
    head_end_indexes = np.cumsum(list(map(len, words))).tolist() + [0]  # 因为root是0,0-1=-1
    for word, head in zip(words, heads):
        char_head = []
        if len(word) > 1:
            char_head.append(char_index + 1)
            char_index += 1
            for _ in range(len(word) - 2):
                char_index += 1
                char_head.append(char_index)
        char_index += 1
        char_head.append(head_end_indexes[head - 1])
        char_heads.extend(char_head)
    return char_heads


# 事先统计的数据信息
global_app_num = 0
app = set()
global_outher_num = 0
outher = set()


def add_char_labels(instance):
    """
    将word_lst中的数据按照下面的方式设置label
    比如"复旦大学 位于 ", 对应的分词是"B M M E B E", 则对应的dependency是"复(dep)->旦(head)", "旦(dep)->大(head)"..
            对应的label是'app', 'app', 'app', , 而学的label就是复旦大学这个词的dependency label
    :param instance:
    :return:
    """
    global global_app_num
    global global_outher_num
    words = instance['word_lst']
    labels = instance['labels']
    pos_tags = instance['pos_tags']
    char_labels = []
    for word, label, pos in zip(words, labels, pos_tags):
        for _ in range(len(word) - 1):
            char_labels.append('APP-' + pos)
            app.add('APP-' + pos)
            global_app_num += 1
        char_labels.append(label + '-' + pos)
        global_outher_num += 1
        outher.add(label + '-' + pos)
    return char_labels


# add seg_targets
def add_segs(instance):
    words = instance['word_lst']
    segs = [0] * len(instance['chars'])
    index = 0
    for word in words:
        index = index + len(word) - 1
        segs[index] = len(word) - 1
        index = index + 1
    return segs


# add target_masks
def add_mask(instance):
    words = instance['word_lst']
    mask = []
    for word in words:
        mask.extend([0] * (len(word) - 1))
        mask.append(1)
    return mask
