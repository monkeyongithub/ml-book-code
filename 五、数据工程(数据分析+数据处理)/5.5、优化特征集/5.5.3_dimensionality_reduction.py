# -*- coding: utf-8 -*-
# 文件用途：比较 PCA、SVD、MDS、t-SNE 和 LDA 对鸢尾花特征的降维效果。
# 对应章节/小节：第 5 章 5.5.3 特征降维
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.5、优化特征集/5.5.3_dimensionality_reduction.py"
# 备注：无监督方法展示三维结果；鸢尾花只有三类，LDA 最多得到二维结果。

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE

RANDOM_STATE = 42


def plot_three_dimensional_methods(features, labels, feature_names, target_names, results):
    """将原始特征和四种无监督降维结果放在两组图中比较。"""
    original = features[:, :3]
    groups = (
        (("原始特征", original), ("PCA", results["PCA"]), ("SVD", results["SVD"])),
        (("原始特征", original), ("t-SNE", results["t-SNE"]), ("MDS", results["MDS"])),
    )
    for group in groups:
        fig = plt.figure(figsize=(15, 5))
        for position, (title, points) in enumerate(group, start=1):
            ax = fig.add_subplot(1, 3, position, projection="3d")
            for class_id, class_name in enumerate(target_names):
                class_points = points[labels == class_id]
                ax.scatter(*class_points.T, label=class_name, s=22)
            ax.set_title(title)
            axes = feature_names[:3] if title == "原始特征" else ("分量 1", "分量 2", "分量 3")
            ax.set_xlabel(axes[0])
            ax.set_ylabel(axes[1])
            ax.set_zlabel(axes[2])
            if position == 1:
                ax.legend()
        fig.tight_layout()
    plt.show()


def plot_lda(original, lda_points, labels, feature_names, target_names):
    """用二维图展示有监督的 LDA，避免把重复坐标误作第三个分量。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for class_id, class_name in enumerate(target_names):
        mask = labels == class_id
        axes[0].scatter(original[mask, 0], original[mask, 1], label=class_name, s=22)
        axes[1].scatter(lda_points[mask, 0], lda_points[mask, 1], label=class_name, s=22)
    axes[0].set(title="原始特征", xlabel=feature_names[0], ylabel=feature_names[1])
    axes[1].set(title="LDA", xlabel="判别分量 1", ylabel="判别分量 2")
    axes[0].legend()
    fig.tight_layout()
    plt.show()


def main():
    # Step1、载入书稿使用的鸢尾花数据；内置数据集无需通过 OpenML 下载。
    iris = load_iris()
    features, labels = iris.data, iris.target

    # Step2、分别计算线性无监督、非线性无监督和线性有监督降维结果。
    # T1、PCA 与 SVD：观察线性变换保留的主要方向。
    results = {
        "PCA": PCA(n_components=3).fit_transform(features),
        "SVD": TruncatedSVD(n_components=3, random_state=RANDOM_STATE).fit_transform(features),
    }
    # T2、MDS 与 t-SNE：观察非线性方法形成的局部或距离结构。
    results["MDS"] = MDS(n_components=3, random_state=RANDOM_STATE).fit_transform(features)
    results["t-SNE"] = TSNE(
        n_components=3, random_state=RANDOM_STATE, init="pca", learning_rate="auto"
    ).fit_transform(features)
    # T3、LDA：利用类别标签寻找判别方向；三类样本最多有两个判别分量。
    lda_points = LinearDiscriminantAnalysis(n_components=2).fit_transform(features, labels)
    for name, points in (*results.items(), ("LDA", lda_points)):
        print(f"{name} 降维后的形状：{points.shape}")

    # Step3、比较原始数据和降维结果；LDA 单独使用二维图表示。
    plot_three_dimensional_methods(
        features, labels, iris.feature_names, iris.target_names, results
    )
    plot_lda(features, lda_points, labels, iris.feature_names, iris.target_names)


if __name__ == "__main__":
    main()
