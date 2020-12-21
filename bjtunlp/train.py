import os
import time
import argparse

from tqdm import tqdm

import torch
from torch import optim
from torch import nn

from fastNLP import BucketSampler
from fastNLP import logger
from fastNLP import DataSetIter
from fastNLP import Tester
from fastNLP import cache_results

from bjtunlp.models import BertParser
from bjtunlp.models.metrics import SegAppCharParseF1Metric, CWSPOSMetric, ParserMetric
from bjtunlp.modules.trianglelr import TriangleLR
from bjtunlp.modules.chart import save_table
from bjtunlp.modules.pipe import CTBxJointPipe
from bjtunlp.modules.word_batch import BatchSampler
from bjtunlp.modules.embedding import ElectraEmbedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str,
                        help=r'Whether to use the first-order model or the second-order model, LOC represents the first-order model CRF2 stands for second-order model. default:LOC',
                        choices=['LOC', 'CRF2'],
                        default='LOC')
    parser.add_argument('--output', type=str,
                        help=r'The path where the output model is stored. default:./output',
                        default=r'output')
    parser.add_argument('--dataset', type=str,
                        help=r'The data set required for training the joint model, which must include the training set, test set and development set, and the data format is CoNLL format. default:./ctb7',
                        default=r'G:\真正联合\bjtunlp\data\ctb7')
    parser.add_argument('--pretraining', type=str,
                        help='Pre-trained language models Electra downloaded from huggingface. default:./discriminator',
                        default=r'H:\预训练语言模型\哈工大20G语料-Electra\base\discriminator')
    parser.add_argument('--epochs', type=int, help='Number of epoch to train the model. default:15', default=15)
    parser.add_argument('--lr', type=float, help='Learning rate setting. default:2e-5', default=2e-5)
    parser.add_argument('--batch_size', type=int, help='The number of words fed to the model at a time. default:1000',
                        default=1000)
    parser.add_argument('--clip', type=float, help='Value for gradient clipping nn.utils.clip_grad_value_. default:5.0',
                        default=5.0)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization. default:1e-2',
                        default=1e-2)
    parser.add_argument('--device', type=int,
                        help='Whether to use GPU for training, 0 means cuda:0, -1 means cpu. default:0',
                        default=0)
    parser.add_argument('--dropout', type=float, help='dropout. default:0.5', default=0.5)
    parser.add_argument('--arc_mlp_size', type=int,
                        help='The hidden dimensions of predicting the dependency arc. default:500',
                        default=500)
    parser.add_argument('--label_mlp_size', type=int,
                        help='The hidden dimensions of predicting the dependency label. default:100',
                        default=300)
    args = parser.parse_args()
    print(args)
    context_path = os.getcwd()
    save_path = os.path.join(context_path, args.output)
    if not os.path.exists(context_path):
        os.makedirs(context_path)

    model_type = args.model_type
    data_name = args.dataset
    pretraining = args.pretraining
    epochs = args.epochs
    lr = args.lr  # 0.01~0.001
    batch_size = args.batch_size  # 1000
    clip = args.clip
    weight_decay = args.weight_decay
    device = torch.device("cuda:%d" % args.device if (torch.cuda.is_available()) else "cpu")
    dropout = args.dropout  # 0.3~0.6
    arc_mlp_size = args.arc_mlp_size  # 200, 300
    label_mlp_size = args.label_mlp_size

    logger.add_file(save_path + '/joint' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log',
                    level='INFO')

    # 将超参数保存的日志中
    logger.info(f'model_type:{model_type}')
    logger.info(f'data_name:{data_name}')
    logger.info(f'pretraining:{pretraining}')
    logger.info(f'epochs:{epochs}')
    logger.info(f'lr:{lr}')
    logger.info(f'batch_size:{batch_size}')
    logger.info(f'clip:{clip}')
    logger.info(f'weight_decay:{weight_decay}')
    logger.info(f'device:{device}')
    logger.info(f'dropout:{dropout}')
    logger.info(f'arc_mlp_size:{arc_mlp_size}')
    logger.info(f'label_mlp_size:{label_mlp_size}')

    cache_name = os.path.split(data_name)[-1]

    @cache_results(save_path + '/caches/{}.pkl'.format(cache_name), _refresh=False)
    def get_data(data_name, pretraining):
        data, special_root = CTBxJointPipe().process_from_file(data_name)
        data.delete_field('bigrams')
        data.delete_field('trigrams')
        data.delete_field('chars')
        data.rename_field('pre_chars', 'chars')
        data.delete_field('pre_bigrams')
        data.delete_field('pre_trigrams')
        embed = ElectraEmbedding(data.get_vocab('chars'), pretraining)
        return data, embed, special_root

    data, embed, special_root = get_data(data_name, pretraining)
    print(data)

    model = BertParser(embed=embed, char_label_vocab=data.get_vocab('char_labels'),
                       num_pos_label=len(data.get_vocab('char_pos')), arc_mlp_size=arc_mlp_size,
                       label_mlp_size=label_mlp_size, dropout=dropout,
                       model=model_type,
                       special_root=special_root,
                       use_greedy_infer=False,
                       )

    metric1 = SegAppCharParseF1Metric(data.get_vocab('char_labels'))
    metric2 = CWSPOSMetric(data.get_vocab('char_labels'), data.get_vocab('char_pos'))
    metric3 = ParserMetric(data.get_vocab('char_labels'))
    metrics = [metric1, metric2, metric3]

    optimizer = optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=lr,
                            weight_decay=weight_decay)

    sampler = BucketSampler(batch_size=4, seq_len_field_name='seq_lens')
    train_batch = DataSetIter(batch_size=4, dataset=data.get_dataset('train'), sampler=sampler,
                              batch_sampler=BatchSampler(data.get_dataset('train'), batch_size, 'seq_lens'))
    scheduler = TriangleLR(optimizer, len(train_batch) * epochs, schedule='linear')
    best_score = 0.
    best_epoch = 0
    table = []
    model = model.to(device)
    for i in range(epochs):
        for batch_x, batch_y in tqdm(train_batch, desc='Epoch: %3d' % i):
            optimizer.zero_grad()
            if args.device >= 0:
                batch_x['chars'] = batch_x['chars'].to(device)
                batch_y['char_heads'] = batch_y['char_heads'].to(device)
                batch_y['char_labels'] = batch_y['char_labels'].to(device)
                batch_y['char_pos'] = batch_y['char_pos'].to(device)
                batch_y['sibs'] = batch_y['sibs'].to(device)

            output = model(batch_x['chars'], batch_y['char_heads'], batch_y['char_labels'],
                           batch_y['sibs'])
            loss = output['loss']
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
        dev_tester = Tester(data.get_dataset('dev'), model, batch_size=8, metrics=metrics, device=device, verbose=0)
        dev_res = dev_tester.test()
        logger.info('Epoch:%3d Dev' % i + dev_tester._format_eval_results(dev_res))
        print('Epoch:%3d Dev' % i + dev_tester._format_eval_results(dev_res))
        test_tester = Tester(data.get_dataset('test'), model, batch_size=8, metrics=metrics, device=device, verbose=0)
        test_res = test_tester.test()
        logger.info('Epoch:%3d Test' % i + test_tester._format_eval_results(test_res))
        print('Epoch:%3d Test' % i + test_tester._format_eval_results(test_res))
        if dev_res['SegAppCharParseF1Metric']['u_f1'] > best_score:
            best_score = dev_res['SegAppCharParseF1Metric']['u_f1']
            best_epoch = i
            torch.save(model, save_path + '/joint.model')
        table.append([dev_res, test_res])

    print('best performance on test dataset Related to the development set %d' % best_epoch)
    print('Save the model in this directory :%s' % save_path)
    logger.info('best performance on test dataset Related to the development set %d' % best_epoch)
    logger.info('Save the model in this directory :%s' % save_path)
    logger.info(str(table[best_epoch]))
    save_table(table, save_path + '/results.csv')


if __name__ == '__main__':
    main()
