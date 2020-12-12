# -*- coding: UTF-8 -*-
"""
-------------------------------------------------
# @Project -> File   ：JointCwsPosParserV2 -> conll
# @Author ：bosskai
# @Date   ：2020/7/30 8:28
# @Email  ：19120406@bjtu.edu.cn
-------------------------------------------------
"""


class CoNLL:
    @staticmethod
    def isprojective(sequence):
        arcs = [(h, d) for d, h in enumerate(sequence[1:], 1) if h >= 0]
        for i, (hi, di) in enumerate(arcs):
            for hj, dj in arcs[i + 1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if (li <= hj <= ri and hi == dj) or (lj <= hi <= rj and hj == di):
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                    return False
        return True
