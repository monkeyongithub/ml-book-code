# -*- coding: utf-8 -*-
# 文件用途：比较信用卡欺诈数据的过采样、欠采样与组合采样方法。
# 对应章节/小节：第 3 章 3.2.5.1—3.2.5.4 数据不均衡与采样方法
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 三、数据科学任务完整流程/3.2、数据认知与数据收集/3.2.5、数据不均衡/3.2.5_sampling_methods.py --method smote
# 备注：仅对训练集采样；部分距离和聚类方法处理完整信用卡数据需要较长时间。

import argparse

import matplotlib.pyplot as plt
import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import (
    ClusterCentroids,
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.dataset import creditcard_df

RANDOM_STATE = 42


def build_sampler(method):
    """按书稿中的采样类别创建所选方法。"""
    samplers = {
        # 3.2.5.2：T1 随机过采样；T2 SMOTE；T3 ADASYN。
        "random_over": RandomOverSampler(random_state=RANDOM_STATE),
        "smote": SMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
        # 3.2.5.3：T1 随机欠采样；T2 NearMiss；T3 聚类中心；
        # T4 Tomek Links；T5 Edited Nearest Neighbours。
        "random_under": RandomUnderSampler(random_state=RANDOM_STATE),
        "near_miss_1": NearMiss(version=1),
        "near_miss_2": NearMiss(version=2),
        "near_miss_3": NearMiss(version=3),
        "cluster_centroids": ClusterCentroids(random_state=RANDOM_STATE),
        "tomek": TomekLinks(),
        "enn": EditedNearestNeighbours(),
        # 3.2.5.4：T1 SMOTE + Tomek；T2 SMOTE + ENN。
        "smote_tomek": SMOTETomek(random_state=RANDOM_STATE),
        "smote_enn": SMOTEENN(random_state=RANDOM_STATE),
    }
    return samplers[method]


def show_class_counts(name, labels):
    """输出两类样本数量，观察采样是否达到预期。"""
    classes, counts = np.unique(labels, return_counts=True)
    print(f"{name}：" + ", ".join(
        f"类别 {class_id} = {count}" for class_id, count in zip(classes, counts)
    ))


def evaluate(features_train, labels_train, features_test, labels_test):
    """使用相同的逻辑回归模型评估一组训练样本。"""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(features_train, labels_train)
    scores = model.predict_proba(features_test)[:, 1]
    return {
        "scores": scores,
        "roc_auc": roc_auc_score(labels_test, scores),
        "average_precision": average_precision_score(labels_test, scores),
    }


def plot_curves(labels_test, results):
    """分别绘制 ROC 与 PR 曲线，避免在同一坐标轴混用两种含义。"""
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for name, result in results.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            labels_test, result["scores"]
        )
        precision, recall, _ = precision_recall_curve(
            labels_test, result["scores"]
        )
        roc_axis.plot(
            false_positive_rate, true_positive_rate,
            label=f"{name} (AUC={result['roc_auc']:.3f})",
        )
        pr_axis.plot(
            recall, precision,
            label=f"{name} (AP={result['average_precision']:.3f})",
        )
    roc_axis.set(xlabel="假阳性率", ylabel="真正率", title="ROC 曲线")
    pr_axis.set(xlabel="召回率", ylabel="精确率", title="PR 曲线")
    roc_axis.legend()
    pr_axis.legend()
    figure.tight_layout()
    plt.show()


def main():
    methods = [
        "random_over", "smote", "adasyn", "random_under", "near_miss_1",
        "near_miss_2", "near_miss_3", "cluster_centroids", "tomek",
        "enn", "smote_tomek", "smote_enn",
    ]
    parser = argparse.ArgumentParser(description="比较原始与采样后的分类效果。")
    parser.add_argument("--method", choices=methods, default="smote")
    method = parser.parse_args().method

    # Step1、载入信用卡数据，先按标签分层切分训练集和测试集。
    data, target_name = creditcard_df()
    features = data.drop(columns=[target_name]).copy()
    features["Time"] = (features["Time"] // 3600).astype(int)
    labels = data[target_name]
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)
    show_class_counts("原始训练集", labels_train)

    # Step2、只对训练集采样；测试集保留原始类别比例。
    sampler = build_sampler(method)
    sampled_features, sampled_labels = sampler.fit_resample(
        features_train, labels_train
    )
    show_class_counts(f"{method} 后的训练集", sampled_labels)

    # Step3、用相同模型分别评估采样前后，在同一测试集比较效果。
    results = {
        "未采样": evaluate(
            features_train, labels_train, features_test, labels_test
        ),
        method: evaluate(
            sampled_features, sampled_labels, features_test, labels_test
        ),
    }
    for name, result in results.items():
        print(
            f"{name}：ROC AUC={result['roc_auc']:.4f}，"
            f"AP={result['average_precision']:.4f}"
        )
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_curves(labels_test, results)


if __name__ == "__main__":
    main()
