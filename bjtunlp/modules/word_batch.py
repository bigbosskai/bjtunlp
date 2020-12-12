# -*- coding: UTF-8 -*-
"""
-------------------------------------------------
# @Project -> File   ：Joint2oCwsPosParser -> word_batch
# @Author ：bosskai
# @Date   ：2020/9/9 11:22
# @Email  ：19120406@bjtu.edu.cn
-------------------------------------------------
"""
import numpy as np


class BatchSampler:
    def __init__(self, data_set, batch_size=3000, seq_len_field_name='seq_len'):
        self.data_set = data_set
        self.batch_size = batch_size
        self.seq_len_field_name = seq_len_field_name

        seq_lens = data_set.get_all_fields()[self.seq_len_field_name].content
        total_sample_num = len(seq_lens)
        sorted_seq_lens = list(sorted([(idx, seq_len) for
                                       idx, seq_len in zip(range(total_sample_num), seq_lens)],
                                      key=lambda x: x[1]))
        self.batches = []
        batch = []
        n_words_per_batch = 0
        for idx, seq_len in sorted_seq_lens:
            batch.append(idx)
            n_words_per_batch += seq_len
            if n_words_per_batch > self.batch_size:
                self.batches.append(batch)
                batch = []
                n_words_per_batch = 0
        if len(batch) != 0:
            self.batches.append(batch)
        np.random.shuffle(self.batches)

    def __iter__(self):
        for batch_idx in self.batches:
            yield batch_idx

    def __len__(self):
        return len(self.batches)
