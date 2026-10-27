# -*- coding: utf-8 -*-
# 文件用途：演示随机切分、分层切分、K 折交叉验证和时间序列切分。
# 对应章节/小节：第 6 章 6.1 数据集划分
# 作者：NYY
# 创建日期：2024-03-22
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.1、数据集划分/6.1_dataset_splitting.py
# 备注：用模拟数据比较四种切分方式；时间序列示例按原顺序划分。

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)


def main():
    # Step1、构造带类别标签的数据，用相同样本比较前三种划分方法。
    features, labels = make_classification(
        n_samples=200, n_features=8, n_informative=5,
        n_redundant=0, weights=[0.75, 0.25], random_state=42,
    )

    # T1、随机留出：适合样本没有时间顺序的基础演示。
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42,
    )
    print(f"随机切分：训练集 {len(x_train)}，测试集 {len(x_test)}")

    # T2、分层留出：保持训练集与测试集的类别比例接近原数据。
    x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    print(
        f"分层切分：训练集正类比例 {y_train_s.mean():.2f}，"
        f"测试集正类比例 {y_test_s.mean():.2f}"
    )

    # T3、K 折交叉验证：分层版本更适合类别不均衡的分类任务。
    model = LogisticRegression(max_iter=500)
    plain_scores = cross_val_score(
        model, features, labels,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    stratified_scores = cross_val_score(
        model, features, labels,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    print(f"KFold 平均准确率：{plain_scores.mean():.3f}")
    print(f"StratifiedKFold 平均准确率：{stratified_scores.mean():.3f}")

    # T4、时间序列切分：每轮只用过去的数据预测更晚的数据。
    ordered_samples = list(range(12))
    splitter = TimeSeriesSplit(n_splits=3)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(ordered_samples), start=1
    ):
        print(
            f"时间序列第 {fold} 折：训练索引 {train_indices.tolist()}，"
            f"测试索引 {test_indices.tolist()}"
        )

    # Step2、比较各方法的索引或类别比例，再按任务选择切分策略。
    print(f"随机留出后的测试集正类比例：{y_test.mean():.2f}")


if __name__ == "__main__":
    main()
