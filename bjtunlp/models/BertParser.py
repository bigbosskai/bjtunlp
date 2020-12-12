from torch import nn
from fastNLP.modules.dropout import TimestepDropout
import torch
from fastNLP import seq_len_to_mask
from fastNLP.models.biaffine_parser import ArcBiaffine, LabelBilinear, BiaffineParser
import torch.nn.functional as F
from bjtunlp.modules.layer import MLP, Triaffine
from bjtunlp.modules.alg import eisner2o, eisner
from bjtunlp.modules.treecrf import CRF2oDependency


class BertParser(BiaffineParser):
    def __init__(self, embed, char_label_vocab, num_pos_label, arc_mlp_size=500, label_mlp_size=100, dropout=0.5,
                 model='CRF', special_root='ROOT', use_greedy_infer=False):
        super(BiaffineParser, self).__init__()
        self.model_type = model
        self.num_pos_label = num_pos_label
        self.embed = embed
        self.char_label_vocab = char_label_vocab
        self.mlp = nn.Sequential(nn.Linear(self.embed.embed_size, arc_mlp_size * 2 + label_mlp_size * 2),
                                 nn.LeakyReLU(0.1),
                                 TimestepDropout(p=dropout), )
        self.crf2 = CRF2oDependency()
        self.special_root = special_root
        if self.model_type == 'CRF2':
            self.mlp_sib_s = MLP(self.embed.embed_size, 100, dropout)
            self.mlp_sib_d = MLP(self.embed.embed_size, 100, dropout)
            self.mlp_sib_h = MLP(self.embed.embed_size, 100, dropout)
            self.sib_attn = Triaffine(n_in=100, bias_x=True, bias_y=True)
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size, len(self.char_label_vocab), bias=True)
        self.use_greedy_infer = use_greedy_infer
        self.reset_parameters()

        self.num_label = len(self.char_label_vocab)

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for name, m in self.named_modules():
            if 'embed' in name:
                pass
            elif hasattr(m, 'reset_parameters') or hasattr(m, 'init_param'):
                pass
            else:
                for p in m.parameters():
                    if len(p.size()) > 1:
                        nn.init.xavier_normal_(p, gain=0.1)
                    else:
                        nn.init.uniform_(p, -0.1, 0.1)

    def _forward(self, chars, gold_heads=None, char_labels=None, sibs=None):
        batch_size, max_len = chars.shape

        feats = self.embed(chars)
        mask = chars.ne(0)

        feats = self.dropout(feats)
        sib_s = None
        if self.model_type == 'CRF2':
            sib_s = self.mlp_sib_s(feats)
            sib_d = self.mlp_sib_d(feats)
            sib_h = self.mlp_sib_h(feats)
            sib_s = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        feats = self.mlp(feats)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feats[:, :, :arc_sz], feats[:, :, arc_sz:2 * arc_sz]
        label_dep, label_head = feats[:, :, 2 * arc_sz:2 * arc_sz + label_sz], feats[:, :, 2 * arc_sz + label_sz:]

        arc_pred = self.arc_predictor(arc_head, arc_dep)  # [N, L, L]

        if gold_heads is None or not self.training:
            # use greedy decoding in training
            if self.training or self.use_greedy_infer:
                heads = self.greedy_decoder(arc_pred, mask)
            else:
                if self.model_type == 'CRF2':
                    mask_ = mask.detach()
                    mask_[:, 0] = 0
                    arc_pred.diagonal(0, 1, 2).fill_(float('-inf'))
                    heads = eisner2o((arc_pred, sib_s), mask_)
                else:
                    mask_ = mask.detach()
                    mask_[:, 0] = 0
                    arc_pred.diagonal(0, 1, 2).fill_(float('-inf'))
                    heads = eisner(arc_pred, mask_)
            head_pred = heads
        else:
            assert self.training  # must be training mode
            if gold_heads is None:
                heads = self.greedy_decoder(arc_pred, mask)
                head_pred = heads
            else:
                head_pred = None
                heads = gold_heads

        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=chars.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep)  # [N, max_len, num_label]

        if gold_heads is not None:
            if self.model_type == 'LOC':
                loss = self.loc_loss(arc_pred, label_pred, gold_heads, char_labels, mask)
            else:
                loss = self.crf2_loss(arc_pred, label_pred, sib_s, gold_heads, char_labels, sibs, mask)
            res_dict = {'loss': loss}
        else:
            res_dict = {'label_preds': label_pred.max(2)[1], 'head_preds': head_pred}
        return res_dict

    def forward(self, chars, char_heads, char_labels, sibs):
        return self._forward(chars, gold_heads=char_heads, char_labels=char_labels, sibs=sibs)

    def loc_loss(self, arc_pred, label_pred, arc_true, label_true, mask):
        """
        Compute loss.

        :param arc_pred: [batch_size, seq_len, seq_len]
        :param label_pred: [batch_size, seq_len, n_tags]
        :param arc_true: [batch_size, seq_len]
        :param label_true: [batch_size, seq_len]
        :param mask: [batch_size, seq_len]
        :return: loss value
        """
        batch_size, seq_len, _ = arc_pred.shape
        flip_mask = (mask == 0)
        _arc_pred = arc_pred.masked_fill(flip_mask.unsqueeze(1), -float('inf'))
        arc_true.data[:, 0].fill_(-1)
        label_true.data[:, 0].fill_(-1)
        arc_nll = F.cross_entropy(_arc_pred.view(-1, seq_len), arc_true.view(-1), ignore_index=-1)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)), label_true.view(-1), ignore_index=-1)
        return arc_nll + label_nll

    def crf2_loss(self, arc_pred, label_pred, s_sib, arc_true, label_true, sibs, mask):

        batch_size, seq_len, _ = arc_pred.shape
        flip_mask = (mask == 0)
        _arc_pred = arc_pred.masked_fill(flip_mask.unsqueeze(1), -float('inf'))

        label_true.data[:, 0].fill_(-1)

        mask_ = (mask == 1)
        mask_[:, 0] = 0
        scores, target = (_arc_pred, s_sib), (arc_true, sibs)
        arc_nll, _ = self.crf2(scores, mask_, target)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)), label_true.view(-1), ignore_index=-1)
        return arc_nll + label_nll

    def predict(self, chars):
        """

        max_len是包含root的

        :param chars: batch_size x max_len
        :return:
        """
        res = self._forward(chars, gold_heads=None)
        return res

    def predict_text(self, text):
        text_input = []
        for sent in text:
            text_input.append(['$'] + sent)

        batch_size = len(text_input)
        feats = self.embed.predict(text_input)
        lens = torch.tensor([len(tokens) for tokens in text_input])
        lens = lens.to(feats.device)
        mask = seq_len_to_mask(lens)

        sib_s = None
        if self.model_type == 'CRF2':
            sib_s = self.mlp_sib_s(feats)
            sib_d = self.mlp_sib_d(feats)
            sib_h = self.mlp_sib_h(feats)
            sib_s = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        feats = self.mlp(feats)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feats[:, :, :arc_sz], feats[:, :, arc_sz:2 * arc_sz]
        label_dep, label_head = feats[:, :, 2 * arc_sz:2 * arc_sz + label_sz], feats[:, :, 2 * arc_sz + label_sz:]

        arc_pred = self.arc_predictor(arc_head, arc_dep)  # [N, L, L]

        if self.model_type == 'CRF2':
            mask_ = mask.detach()
            mask_[:, 0] = 0
            arc_pred.diagonal(0, 1, 2).fill_(float('-inf'))
            heads = eisner2o((arc_pred, sib_s), mask_)
        else:
            mask_ = mask.detach()
            mask_[:, 0] = 0
            arc_pred.diagonal(0, 1, 2).fill_(float('-inf'))
            heads = eisner(arc_pred, mask_)

        head_pred = heads
        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=feats.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep)

        label_preds = label_pred.max(2)[1]
        head_preds = head_pred

        head_preds = head_preds[:, 1:].tolist()
        label_preds = label_preds[:, 1:].tolist()
        seq_lens = (lens - 1).tolist()
        ret_words = []
        ret_poss = []
        ret_heads = []
        ret_labels = []

        for b in range(batch_size):
            seq_len = seq_lens[b]
            head_pred = head_preds[b][:seq_len]
            label_pred = label_preds[b][:seq_len]
            char_seqs = text[b]
            words = []
            heads = []
            poss = []
            labels = []
            ranges = []
            word_idx = 0
            word_start_idx = 0
            for idx, (label, head) in enumerate(zip(label_pred, head_pred)):
                ranges.append(word_idx)
                tag = self.char_label_vocab.idx2word[label]
                dep_tag, pos_tag = tag.split('-')
                if dep_tag == 'APP' and idx != seq_len - 1:  # 如果依存标签前面三个字符是APP的话
                    pass
                else:
                    poss.append(pos_tag)
                    if dep_tag == 'APP' and idx == seq_len - 1:
                        labels.append(labels[-1])  # 重复上面一个的依存标签
                    else:
                        labels.append(dep_tag)
                    heads.append(head)
                    words.append(''.join(char_seqs[word_start_idx:idx + 1]))
                    word_start_idx = idx + 1
                    word_idx += 1
            ret_words.append(words)
            ret_poss.append(poss)
            ret_labels.append(labels)
            recover_heads = []
            for i, h in enumerate(heads):
                if h == 0:
                    recover_heads.append(h)
                    labels[i] = self.special_root
                elif h - 1 < len(ranges):
                    recover_heads.append(ranges[h - 1] + 1)
                else:
                    recover_heads.append(ranges[-1])
            ret_heads.append(recover_heads)

        return ret_words, ret_poss, ret_heads, ret_labels
