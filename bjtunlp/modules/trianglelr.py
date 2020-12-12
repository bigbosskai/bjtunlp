# -*- coding: UTF-8 -*-
"""
-------------------------------------------------
# @Project -> File   ：Joint2oCwsPosParser -> trianglelr
# @Author ：bosskai
# @Date   ：2020/9/12 19:25
# @Email  ：19120406@bjtu.edu.cn
-------------------------------------------------
"""


class TriangleLR:
    r"""
    learning rate按照一定的速率从0上升到设置的learning rate。
    """

    def __init__(self, optimizer, total_steps, warmup=0.1, schedule='constant'):
        """

        :param optimizer: 传入优化器
        :param total_steps: 总共的batch数，例如数据集有123个batch，训练的轮数为100，该值设置为12300
        :param warmup: float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param schedule: str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.

        """
        self.optimizer = optimizer
        self.t_steps = total_steps
        self.warmup = max(warmup, 0.)
        self.current_step = 1
        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")
        self.init()

    def _get_constant_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def init(self):
        if self.warmup > 1:
            self.warmup = self.warmup / self.t_steps
        self.t_steps = max(2, self.t_steps)  # 不能小于2
        # 获取param_group的初始learning rate
        for group in self.optimizer.param_groups:
            self.initial_lrs.append(group['lr'])

    def step(self):
        progress = self.current_step / self.t_steps
        for lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
            group['lr'] = lr * self.get_lr(progress)
        self.current_step += 1
