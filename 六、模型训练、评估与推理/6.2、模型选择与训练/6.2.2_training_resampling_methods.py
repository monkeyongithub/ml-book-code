# -*- coding: utf-8 -*-
# 文件用途：比较留出、重复分层划分和自助采样三种模型训练评估方法。
# 对应章节/小节：第 6 章 6.2.2 模型训练
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.2、模型选择与训练/6.2.2_training_resampling_methods.py
# 备注：用同一合成分类数据比较方法；自助采样只在训练集内进行。

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def main():
    # Step1、生成分类样本，固定一份测试集供最终评价。
    features, labels = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=0, weights=[0.7, 0.3], random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    base_model = LogisticRegression(max_iter=500)

    # T1、留出法：在训练集拟合一次，最终再用独立测试集评价。
    holdout_model = clone(base_model).fit(x_train, y_train)

    # T2、重复分层划分：在训练数据内部反复划分，观察分数波动。
    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=42,
    )
    validation_scores = []
    for fit_indices, valid_indices in splitter.split(x_train, y_train):
        model = clone(base_model).fit(
            x_train[fit_indices], y_train[fit_indices]
        )
        score = accuracy_score(
            y_train[valid_indices], model.predict(x_train[valid_indices])
        )
        validation_scores.append(score)
    print(f"重复分层验证平均准确率：{np.mean(validation_scores):.3f}")

    # T3、自助法：有放回地抽取训练样本，用袋外样本估计分数波动。
    generator = np.random.default_rng(42)
    bootstrap_scores = []
    for _ in range(20):
        sample_indices = generator.choice(
            len(x_train), size=len(x_train), replace=True
        )
        if len(np.unique(y_train[sample_indices])) < 2:
            continue
        out_of_bag = np.setdiff1d(
            np.arange(len(x_train)), np.unique(sample_indices)
        )
        if not len(out_of_bag):
            continue
        model = clone(base_model).fit(
            x_train[sample_indices], y_train[sample_indices]
        )
        bootstrap_scores.append(
            accuracy_score(
                y_train[out_of_bag], model.predict(x_train[out_of_bag])
            )
        )
    if not bootstrap_scores:
        raise RuntimeError("自助采样未得到可评价的袋外样本。")

    # Step2、比较训练集内部的验证结果；测试集仅作一次最终评价。
    print(f"自助采样袋外准确率均值：{np.mean(bootstrap_scores):.3f}")
    print(f"自助采样袋外准确率标准差：{np.std(bootstrap_scores):.3f}")
    holdout_score = accuracy_score(y_test, holdout_model.predict(x_test))
    print(f"留出法最终测试准确率：{holdout_score:.3f}")


if __name__ == "__main__":
    main()
