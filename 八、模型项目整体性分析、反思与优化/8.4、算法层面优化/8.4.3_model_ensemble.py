# -*- coding: utf-8 -*-
# 文件用途：演示分类投票和回归预测的多模型融合。
# 对应章节/小节：第 8 章 8.4.3 多算法模型融合
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.4、算法层面优化/8.4.3_model_ensemble.py"。
# 备注：分类软投票使用概率而非类别编号；回归权重从独立验证集估计。

import numpy as np
from sklearn.metrics import r2_score


def normalized_inverse_errors(errors):
    weights = 1 / np.maximum(errors, 1e-12)
    return weights / weights.sum()


def main():
    # Step1、在相同分类样本上展示等权硬投票与加权软投票。
    # 两组同时改变投票形式和权重，输出差异不能只归因于硬/软投票。
    hard_labels = np.array(
        [[0, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1]]
    )
    positive_probabilities = np.array(
        [[0.20, 0.80, 0.75, 0.30],
         [0.65, 0.70, 0.40, 0.25],
         [0.10, 0.85, 0.80, 0.70]]
    )
    classifier_weights = np.array([0.2, 0.3, 0.5])
    # T1、硬投票按类别计数。
    hard_vote = (hard_labels.mean(axis=0) >= 0.5).astype(int)
    # T2、软投票加权平均正类概率。
    soft_probability = np.average(
        positive_probabilities, axis=0, weights=classifier_weights
    )
    soft_vote = (soft_probability >= 0.5).astype(int)
    print("硬投票类别：", hard_vote)
    print("软投票正类概率：", soft_probability)
    print("软投票类别：", soft_vote)

    # Step2、准备同一组回归验证预测与新样本预测，计算各模型权重。
    valid_truth = np.array([2.0, 4.0, 6.0, 8.0])
    valid_predictions = np.array(
        [[2.2, 4.1, 5.7, 7.8],
         [1.8, 4.4, 6.2, 8.1],
         [2.3, 3.7, 6.4, 7.9]]
    )
    new_predictions = np.array(
        [[3.0, 5.0], [3.2, 4.8], [2.9, 5.1]]
    )
    mae = np.mean(np.abs(valid_predictions - valid_truth), axis=1)
    mse = np.mean((valid_predictions - valid_truth) ** 2, axis=1)
    r2 = np.array(
        [r2_score(valid_truth, row) for row in valid_predictions]
    )
    mae_weights = normalized_inverse_errors(mae)
    mse_weights = normalized_inverse_errors(mse)
    r2_weights = np.maximum(r2, 0)
    if r2_weights.sum() == 0:
        r2_weights = np.ones_like(r2_weights)
    r2_weights = r2_weights / r2_weights.sum()

    # Step3、在同一组新样本预测上比较直接融合与加权融合。
    # T1、等权平均不使用验证误差。
    print("平均融合：", new_predictions.mean(axis=0))
    # T2、中位数融合不使用验证误差。
    print("中位数融合：", np.median(new_predictions, axis=0))

    # T3、MAE 权重；T4、MSE 权重；T5、R² 权重：只改变权重依据。
    for name, weights in (
        ("MAE 权重", mae_weights),
        ("MSE 权重", mse_weights),
        ("R² 权重", r2_weights),
    ):
        print(f"{name}：", np.average(new_predictions, axis=0, weights=weights))


if __name__ == "__main__":
    main()
