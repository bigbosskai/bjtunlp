from collections import Counter
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import seq_len_to_mask
import torch


class SegAppCharParseF1Metric(MetricBase):
    #
    def __init__(self, char_label_vocab):
        super().__init__()
        self.char_label_vocab = char_label_vocab

        self.parse_head_tp = 0
        self.parse_label_tp = 0
        self.rec_tol = 0
        self.pre_tol = 0
        self.sent_tol = 0
        self.u_sent = 0
        self.l_sent = 0

    def evaluate(self, gold_word_pairs, gold_label_word_pairs, head_preds, label_preds, seq_lens,
                 pun_masks):
        """

        max_len是不包含root的character的长度
        :param gold_word_pairs: List[List[((head_start, head_end), (dep_start, dep_end)), ...]], batch_size
        :param gold_label_word_pairs: List[List[((head_start, head_end), label, (dep_start, dep_end)), ...]], batch_size
        :param head_preds: batch_size x max_len
        :param label_preds: batch_size x max_len
        :param seq_lens:
        :param pun_masks: batch_size x
        :return:
        """
        # 去掉root
        # head_preds = char_heads
        # label_preds = char_labels
        head_preds = head_preds[:, 1:].tolist()
        label_preds = label_preds[:, 1:].tolist()
        seq_lens = (seq_lens - 1).tolist()
        self.sent_tol = self.sent_tol + len(seq_lens)
        # 先解码出words，POS，heads, labels, 对应的character范围
        for b in range(len(head_preds)):
            seq_len = seq_lens[b]
            head_pred = head_preds[b][:seq_len]
            label_pred = label_preds[b][:seq_len]

            words = []  # 存放[word_start, word_end)，相对起始位置，不考虑root
            heads = []
            labels = []
            ranges = []  # 对应该char是第几个word，长度是seq_len+1
            word_idx = 0
            word_start_idx = 0
            for idx, (label, head) in enumerate(zip(label_pred, head_pred)):
                ranges.append(word_idx)
                tag = self.char_label_vocab.idx2word[label]
                # if tag == '<unk>':
                #     tag = 'nn-NN'
                dep_tag, pos_tag = tag.split('-')
                if dep_tag == 'APP':  # 如果依存标签前面三个字符是APP的话
                    pass
                else:

                    labels.append(dep_tag)
                    heads.append(head)
                    words.append((word_start_idx, idx + 1))
                    word_start_idx = idx + 1
                    word_idx += 1

            head_dep_tuple = []  # head在前面
            head_label_dep_tuple = []
            for idx, head in enumerate(heads):
                span = words[idx]
                if span[0] == span[1] - 1 and pun_masks[b, span[0]]:
                    continue  # exclude punctuations
                if head == 0:
                    head_dep_tuple.append((('root', words[idx])))
                    head_label_dep_tuple.append(('root', labels[idx], words[idx]))
                else:
                    head_word_idx = ranges[head - 1]
                    head_word_span = words[head_word_idx] if head_word_idx < len(words) else words[0]
                    head_dep_tuple.append(((head_word_span, words[idx])))
                    head_label_dep_tuple.append((head_word_span, labels[idx], words[idx]))
            gold_head_dep_tuple = set([(tuple(pair[0]) if not isinstance(pair[0], str) else pair[0],
                                        tuple(pair[1]) if not isinstance(pair[1], str) else pair[1]) for pair in
                                       gold_word_pairs[b]])
            gold_head_label_dep_tuple = set([(tuple(pair[0]) if not isinstance(pair[0], str) else pair[0],
                                              pair[1],
                                              tuple(pair[2]) if not isinstance(pair[2], str) else pair[2]) for pair in
                                             gold_label_word_pairs[b]])

            u_flag = True
            l_flag = True
            for head_dep, head_label_dep in zip(head_dep_tuple, head_label_dep_tuple):
                if head_dep in gold_head_dep_tuple:
                    self.parse_head_tp += 1
                else:
                    u_flag = False
                if head_label_dep in gold_head_label_dep_tuple:
                    self.parse_label_tp += 1
                else:
                    l_flag = False
            self.pre_tol += len(head_dep_tuple)
            self.rec_tol += len(gold_head_dep_tuple)
            if u_flag:
                self.u_sent = self.u_sent + 1
            if l_flag:
                self.l_sent = self.l_sent + 1

    def get_metric(self, reset=True):
        u_p = self.parse_head_tp / (1e-6 + self.pre_tol)
        u_r = self.parse_head_tp / (1e-6 + self.rec_tol)
        u_f = 2 * u_p * u_r / (1e-6 + u_p + u_r)
        l_p = self.parse_label_tp / (1e-6 + self.pre_tol)
        l_r = self.parse_label_tp / (1e-6 + self.rec_tol)
        l_f = 2 * l_p * l_r / (1e-6 + l_p + l_r)
        lcm = self.l_sent / (1e-6 + self.sent_tol)
        ucm = self.u_sent / (1e-6 + self.sent_tol)
        if reset:
            self.parse_head_tp = 0
            self.parse_label_tp = 0
            self.rec_tol = 0
            self.pre_tol = 0
            self.sent_tol = 0
            self.u_sent = 0
            self.l_sent = 0

        return {'u_f1': round(u_f, 4), 'u_p': round(u_p, 4), 'u_r/uas': round(u_r, 4),
                'l_f1': round(l_f, 4), 'l_p': round(l_p, 4), 'l_r/las': round(l_r, 4), 'ucm': round(ucm, 4),
                'lcm': round(lcm, 4)}


class CWSPOSMetric(MetricBase):

    def __init__(self, char_label_vocab, char_pos_vocab):
        super().__init__()
        self.char_label_vocab = char_label_vocab
        self.char_pos_vocab = char_pos_vocab
        self.pre = 0
        self.rec = 0
        self.cws_tp = 0
        self.pos_tp = 0

    def evaluate(self, seg_targets, seg_masks, label_preds, char_pos, seq_lens):
        """

        :param seg_targets: batch_size x max_len, 每个位置预测的是该word的长度-1，在word结束的地方。 [0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 3]
        :param seg_masks: batch_size x max_len，只有在word结束的地方为1 [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        :param label_preds: batch_size x max_len
        :param char_pos: batch_size x max_len
        :param seq_lens: batch_size
        :return:
        """
        # label_preds = char_labels
        char_pos = char_pos[:, 1:]

        pred_masks = torch.zeros_like(seg_masks)
        pred_segs = torch.zeros_like(seg_targets)
        pred_pos = torch.zeros_like(char_pos)  # 用于预测的词性标签

        seq_lens = (seq_lens - 1).tolist()  # 去除root，得到句子真实的句子长度
        pos_counter = Counter()
        for idx, label_pred in enumerate(label_preds[:, 1:].tolist()):
            # 表示第0句话
            seq_len = seq_lens[idx]  # 这句话的长度为seq_len
            label_pred = label_pred[:seq_len]  # 预测的依存标签为label_pred这个列表

            word_len = 0
            pos_tag = 'NN'  # 初始化随机给一个词性
            for l_i, label in enumerate(label_pred):
                tag = self.char_label_vocab.idx2word[label]
                # if tag == '<unk>':
                #     tag = 'nn-NN'
                dep_tag, pos_tag = tag.split('-')
                if dep_tag == 'APP' and l_i != len(label_pred) - 1:  # label必须是APP 同时当前不能是最后一个位置，因为最后一个位置不可能是APP
                    word_len += 1
                    pos_counter[pos_tag] += 1
                else:
                    pos_counter[pos_tag] += 1
                    pred_segs[idx, l_i] = word_len  # 这个词的长度为word_len
                    pred_masks[idx, l_i] = 1
                    p = pos_counter.most_common(1)[0][0]
                    pred_pos[idx, l_i] = self.char_pos_vocab.word2idx[p]
                    word_len = 0
                    pos_counter.clear()

        right_mask = seg_targets.eq(pred_segs)  # 对长度的预测一致，表示长度一直
        pos_right_mask = char_pos.eq(pred_pos)
        self.rec += seg_masks.sum().item()
        self.pre += pred_masks.sum().item()
        pred_masks_ = (pred_masks == 1)
        seg_masks_ = (seg_masks == 1)
        # 且pred和target在同一个地方有值
        cws_tp_mask = right_mask.__and__(pred_masks_.__and__(seg_masks_))
        pos_tp_mask = cws_tp_mask.__and__(pos_right_mask)
        self.cws_tp += cws_tp_mask.sum().item()  # 表示长度一直，并且在同一个位置有值
        self.pos_tp += pos_tp_mask.sum().item()

    def get_metric(self, reset=True):
        res = {}
        res['cws_rec'] = round(self.cws_tp / (self.rec + 1e-6), 4)
        res['cws_pre'] = round(self.cws_tp / (self.pre + 1e-6), 4)
        res['cws_f1'] = round(2 * res['cws_rec'] * res['cws_pre'] / (res['cws_rec'] + res['cws_pre'] + 1e-6), 4)
        res['pos_rec'] = round(self.pos_tp / (self.rec + 1e-6), 4)
        res['pos_pre'] = round(self.pos_tp / (self.pre + 1e-6), 4)
        res['pos_f1'] = round(2 * res['pos_rec'] * res['pos_pre'] / (res['pos_rec'] + res['pos_pre'] + 1e-6), 4)

        if reset:
            self.pre = 0
            self.rec = 0
            self.cws_tp = 0
            self.pos_tp = 0

        return res


class ParserMetric(MetricBase):
    def __init__(self, char_label_vocab):
        super().__init__()
        self.num_arc = 0
        self.num_label = 0
        self.num_sample = 0
        self.tol_sent = 0
        self.num_sent = 0
        self.num_sent_label = 0
        self.app_tol = 0
        self.num_app = 0
        self.char_label_vocab = char_label_vocab

    # gold_word_pairs, gold_label_word_pairs, head_preds, label_preds, seq_lens,pos_tags,
    #                  pun_masks):
    def get_metric(self, reset=True):
        res = {'UAS': round(self.num_arc * 1.0 / self.num_sample, 4),
               'LAS': round(self.num_label * 1.0 / self.num_sample, 4),
               'UCM': round(self.num_sent * 1.0 / self.tol_sent, 4),
               'LCM': round(self.num_sent_label * 1.0 / self.tol_sent, 4),
               'AH': round(self.num_app * 1.0 / self.app_tol, 4)}
        if reset:
            self.num_arc = 0
            self.num_label = 0
            self.num_sample = 0
            self.tol_sent = 0
            self.num_sent = 0
            self.num_sent_label = 0
            self.app_tol = 0
            self.num_app = 0
        return res

    def evaluate(self, head_preds, label_preds, char_heads, char_labels, seq_lens):
        """Evaluate the performance of prediction.
           写下对光秃秃的树，精度分析的结果看看依存弧预测精度如何
        """
        # mask out <root> tag
        seq_lens = (seq_lens - 1).tolist()
        head_preds = head_preds[:, 1:].tolist()
        label_preds = label_preds[:, 1:].tolist()
        char_heads = char_heads[:, 1:].tolist()
        char_labels = char_labels[:, 1:].tolist()
        self.tol_sent += len(seq_lens)  # 有多少个句子加起来
        for b, seq_len in enumerate(seq_lens):
            head_pred = head_preds[b][:seq_len]
            label_pred = label_preds[b][:seq_len]
            char_head = char_heads[b][:seq_len]
            char_label = char_labels[b][:seq_len]
            u_flag = True
            l_flag = True
            for h_pred, h_gold, l_pred, l_gold in zip(head_pred, char_head, label_pred, char_label):
                tag_gold = self.char_label_vocab.idx2word[l_gold]
                if 'APP' in tag_gold:
                    self.app_tol += 1
                    if h_pred == h_gold:
                        self.num_app += 1

                if h_pred == h_gold:  # 预测的弧对了一个
                    self.num_arc += 1
                    if l_pred == l_gold:
                        self.num_label += 1
                    else:
                        l_flag = False
                else:
                    u_flag = False
                    l_flag = False
                self.num_sample += 1  # 弧的个数加1

            if u_flag:
                self.num_sent += 1
                if l_flag:
                    self.num_sent_label += 1
