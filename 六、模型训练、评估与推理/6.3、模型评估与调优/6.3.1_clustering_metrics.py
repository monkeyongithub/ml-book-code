# -*- coding: utf-8 -*-
# 文件用途：以 Iris 聚类示例计算 SSE、轮廓系数和 ARI，并比较不同簇数。
# 对应章节/小节：第 6 章 6.3.1 模型评估：聚类任务
# 作者：NYY
# 创建日期：2024-08-14
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.3、模型评估与调优/6.3.1_clustering_metrics.py
# 备注：ARI 使用已知类别仅作事后评价，不参与 KMeans 训练。

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from shared.dataset import iris_df


def main():
    # Step1、读取 Iris 数据；标准化数值特征，避免量纲影响距离。
    frame, target_name = iris_df()
    feature_names = frame.drop(columns=target_name).select_dtypes(
        include="number"
    ).columns.tolist()
    if len(feature_names) < 3:
        raise ValueError("三维聚类图至少需要三个数值特征。")
    features = StandardScaler().fit_transform(frame[feature_names])
    true_labels = frame[target_name]

    # Step2、只用特征训练 KMeans，再计算三类评估指标。
    model = KMeans(n_clusters=3, n_init=10, random_state=42)
    predicted = model.fit_predict(features)
    print(f"SSE：{model.inertia_:.3f}")
    print(f"轮廓系数：{silhouette_score(features, predicted):.3f}")
    print(f"ARI：{adjusted_rand_score(true_labels, predicted):.3f}")

    # Step3、比较簇数的 SSE 与轮廓系数，并查看前三个特征的分组。
    candidate_counts = range(2, 9)
    sse_values, silhouette_values = [], []
    for count in candidate_counts:
        labels = KMeans(
            n_clusters=count, n_init=10, random_state=42
        ).fit_predict(features)
        centers = np.array([
            features[labels == index].mean(axis=0)
            for index in range(count)
        ])
        sse_values.append(np.sum((features - centers[labels]) ** 2))
        silhouette_values.append(silhouette_score(features, labels))

    figure = plt.figure(figsize=(13, 4))
    first = figure.add_subplot(131)
    first.plot(candidate_counts, sse_values, marker="o")
    first.set(xlabel="簇数", ylabel="SSE", title="肘部法")
    second = figure.add_subplot(132)
    second.plot(candidate_counts, silhouette_values, marker="o")
    second.set(xlabel="簇数", ylabel="轮廓系数", title="轮廓系数比较")
    third = figure.add_subplot(133, projection="3d")
    third.scatter(features[:, 0], features[:, 1], features[:, 2], c=predicted)
    third.set(
        xlabel=feature_names[0], ylabel=feature_names[1],
        zlabel=feature_names[2], title="前三个特征的聚类结果",
    )
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
