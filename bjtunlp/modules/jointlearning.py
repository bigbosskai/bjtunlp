# -*- coding: UTF-8 -*-
"""
-------------------------------------------------
# @Project -> File   ：Joint2oCwsPosParser -> jointlearning
# @Author ：bosskai
# @Date   ：2020/9/12 10:02
# @Email  ：19120406@bjtu.edu.cn
-------------------------------------------------
"""
import torch
import torch.nn as nn


class JointLearn(nn.Module):
    def __init__(self, num_tasks):
        """

        :param num_tasks:the number of tasks
        """
        super(JointLearn, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *args):
        """

        :param args:each loss for multi tasks ,for example, there are three tasks ,loss1,loss2,loss3
        :return: joint loss for multi tasks
        """
        losses = []
        for i, L in enumerate(args):
            precision = torch.exp(-self.log_vars[i])
            losses.append(precision * L + self.log_vars[i])
        return sum(losses)
