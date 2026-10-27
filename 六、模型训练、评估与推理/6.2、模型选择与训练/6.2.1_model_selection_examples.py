# -*- coding: utf-8 -*-
# 文件用途：用分类、回归与聚类任务说明模型选择和基础训练流程。
# 对应章节/小节：第 6 章 6.2.1 选择算法（偏实战应用）
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.2、模型选择与训练/6.2.1_model_selection_examples.py
# 备注：使用模拟数据，不依赖外部数据集；聚类标签只用于观察结构。

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


def show_classification():
    # 二分类示例：用逻辑回归预测两个类别。
    features, labels = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=0, random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    model = LogisticRegression(max_iter=500).fit(x_train, y_train)
    print(f"二分类准确率：{accuracy_score(y_test, model.predict(x_test)):.3f}")

    # 多分类示例：用随机森林预测三个类别，说明类别数变化后的处理方式。
    features, labels = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=0, n_classes=3, n_clusters_per_class=1,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    print(f"多分类准确率：{accuracy_score(y_test, model.predict(x_test)):.3f}")


def show_regression():
    # 回归示例：连续目标值使用线性回归预测，并用均方误差评价。
    features, targets = make_regression(
        n_samples=300, n_features=8, noise=10, random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42,
    )
    model = LinearRegression().fit(x_train, y_train)
    print(f"回归均方误差：{mean_squared_error(y_test, model.predict(x_test)):.3f}")

    # 用交叉验证观察 Ridge 正则化强度对误差的影响。
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    errors = [
        -cross_val_score(
            Ridge(alpha=alpha), x_train, y_train,
            cv=5, scoring="neg_mean_squared_error",
        ).mean()
        for alpha in alphas
    ]
    plt.semilogx(alphas, errors, marker="o")
    plt.xlabel("Ridge alpha")
    plt.ylabel("交叉验证均方误差")
    plt.title("正则化强度与回归误差")
    plt.tight_layout()
    plt.show()


def show_clustering():
    # 结构化样本示例：按二维样本的距离寻找四个簇。
    features, _ = make_blobs(
        n_samples=200, centers=4, cluster_std=1.0, random_state=42,
    )
    model = KMeans(n_clusters=4, n_init=10, random_state=42).fit(features)
    plt.scatter(features[:, 0], features[:, 1], c=model.labels_, cmap="viridis")
    plt.scatter(
        model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
        marker="x", color="red", label="簇中心",
    )
    plt.legend()
    plt.title("结构化样本聚类")
    plt.show()

    # 文本示例：先转为词袋向量，再观察聚类标签。
    documents = [
        "data visualization explains patterns",
        "visualization shows data trends",
        "machine learning trains models",
        "models learn from examples",
    ]
    vectors = CountVectorizer().fit_transform(documents)
    text_model = KMeans(n_clusters=2, n_init=10, random_state=42)
    print(f"文本聚类标签：{text_model.fit_predict(vectors).tolist()}")


def main():
    # Step1、阅读分类任务：分别观察二分类和多分类的建模流程。
    show_classification()
    # Step2、阅读回归任务：预测连续目标值并比较误差。
    show_regression()
    # Step3、阅读无监督任务：比较结构化样本与文本的聚类输入。
    show_clustering()


if __name__ == "__main__":
    main()
