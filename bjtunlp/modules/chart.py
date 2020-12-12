# -*- coding: UTF-8 -*-
# @Author ：Kai Wang
# @Date   ：2020/9/9 13:03
# @Email  ：425065178@qq.com/19120406@bjtu.edu.cn

import pandas as pd


def save_table(table, filename):
    dev = []
    test = []
    for dev_res, test_res in table:
        dev_seg_parser_metric = {'dev_' + key: val for key, val in dev_res['SegAppCharParseF1Metric'].items()}
        dev_cws_pos = {'dev_' + key: val for key, val in dev_res['CWSPOSMetric'].items()}
        dev.append({**dev_seg_parser_metric, **dev_cws_pos})

        test_seg_parser_metric = {'test_' + key: val for key, val in test_res['SegAppCharParseF1Metric'].items()}
        test_cws_pos = {'test_' + key: val for key, val in test_res['CWSPOSMetric'].items()}
        test.append({**test_seg_parser_metric, **test_cws_pos})
    df = pd.DataFrame()
    for de, te in zip(dev, test):
        m = {**de, **te}
        ks = []
        vs = []
        for k, v in m.items():
            ks.append(k)
            vs.append(v)
        s = pd.Series(vs, ks)
        df = df.append(s, ignore_index=True)
    # df.to_excel(filename)
    df.to_csv(filename)
