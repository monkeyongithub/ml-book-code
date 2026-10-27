# -*- coding: utf-8 -*-
# 文件用途：比较过滤式、包裹式和嵌入式三类特征筛选方法。
# 对应章节/小节：第 5 章 5.5.2 特征筛选
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.5、优化特征集/5.5.2_feature_selection.py"
# 备注：填补、编码与筛选仅在训练集拟合；验证集只用于包裹式方法的特征数比较。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.dataset import titanic_df

RANDOM_STATE = 42


def prepare_features():
    """先划分原始数据，再从训练集学习缺失值填补和类别编码规则。"""
    frame, label_name = titanic_df()
    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]
    raw_features = frame[numeric + categorical].copy()
    labels = frame[label_name].astype(int)
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        raw_features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )
    train_features = train_features.copy()
    valid_features = valid_features.copy()

    medians = train_features[numeric].median()
    train_features[numeric] = train_features[numeric].fillna(medians)
    valid_features[numeric] = valid_features[numeric].fillna(medians)
    for column in categorical:
        modes = train_features[column].mode(dropna=True)
        fill_value = modes.iloc[0] if not modes.empty else "未知"
        train_features[column] = train_features[column].fillna(fill_value)
        valid_features[column] = valid_features[column].fillna(fill_value)

    train_encoded = pd.get_dummies(train_features, columns=categorical, dtype=int)
    valid_encoded = pd.get_dummies(valid_features, columns=categorical, dtype=int)
    valid_encoded = valid_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, valid_encoded, train_labels, valid_labels


def show_filter_methods(features, labels):
    """用特征自身分布、相关性和统计评分筛选候选列。"""
    # T1、过滤式：先比较方差，再查看特征相关性与标签关联程度。
    variances = features.var().sort_values(ascending=False)
    print("方差排序：", variances)
    print("低方差候选：", variances[variances <= 0.2].index.tolist())

    comparison = features.assign(target=labels)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, method, title in zip(
        axes, ("pearson", "spearman"), ("皮尔逊相关系数", "斯皮尔曼等级相关系数")
    ):
        sns.heatmap(comparison.corr(method=method), ax=ax, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title)
    fig.tight_layout()
    plt.show()

    # 卡方只适用于非负特征；本例的原始数值列与独热编码列均满足要求。
    best_chi2 = SelectKBest(chi2, k=min(5, features.shape[1])).fit(features, labels)
    print("卡方筛选：", features.columns[best_chi2.get_support()].tolist())
    chi2_values, chi2_pvalues = chi2(features, labels)
    f_values, f_pvalues = f_classif(features, labels)
    mi_values = mutual_info_classif(features, labels, random_state=RANDOM_STATE)
    scores = pd.DataFrame({
        "卡方值": chi2_values,
        "卡方 p 值": chi2_pvalues,
        "F 值": f_values,
        "F 检验 p 值": f_pvalues,
        "互信息": mi_values,
    }, index=features.columns)
    print("过滤式评分：", scores)
    return scores


def show_wrapper_methods(features, labels, valid_features, valid_labels):
    """仅在训练集拟合缩放器与 RFE，用验证集比较特征数。"""
    # T2、包裹式：验证集只用于比较特征数，不参与填补、编码或缩放拟合。
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features)
    valid_scaled = scaler.transform(valid_features)
    errors, selections = [], []
    counts = range(1, features.shape[1] + 1)
    for count in counts:
        selector = RFE(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            n_features_to_select=count,
        )
        selector.fit(train_scaled, labels)
        predictions = selector.predict(valid_scaled)
        errors.append(1 - accuracy_score(valid_labels, predictions))
        selections.append(selector.support_)

    best_count = int(np.argmin(errors)) + 1
    print("验证误差最低时的特征数：", best_count)
    print("对应特征：", features.columns[selections[best_count - 1]].tolist())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(counts), errors, marker="o")
    axes[0].set(xlabel="选择的特征数量", ylabel="验证误差", title="RFE 特征数与误差")
    sns.heatmap(np.array(selections).T, ax=axes[1], cmap="Blues", cbar=False)
    axes[1].set(xlabel="选择的特征数量", ylabel="特征", title="RFE 选择路径")
    axes[1].set_yticks(np.arange(len(features.columns)) + 0.5)
    axes[1].set_yticklabels(features.columns, rotation=0)
    fig.tight_layout()
    plt.show()


def show_embedded_methods(features, labels):
    """比较线性模型系数和树模型重要性所选的特征。"""
    # T3、嵌入式：筛选依据来自模型训练过程。
    scaled = StandardScaler().fit_transform(features)
    logistic_selector = SelectFromModel(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    ).fit(scaled, labels)
    print("逻辑回归选择：", features.columns[logistic_selector.get_support()].tolist())

    from lightgbm import LGBMClassifier

    tree_selector = SelectFromModel(
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    ).fit(features, labels)
    print("LightGBM 选择：", features.columns[tree_selector.get_support()].tolist())


def main():
    # Step1、先划分 Titanic 数据，再从训练集学习填补和编码规则。
    features, valid_features, labels, valid_labels = prepare_features()
    print("候选特征：", features.columns.tolist())

    # Step2、在训练数据上比较书稿中的三大类特征筛选方法。
    show_filter_methods(features, labels)
    show_wrapper_methods(features, labels, valid_features, valid_labels)
    show_embedded_methods(features, labels)

    # Step3、结合验证结果与业务含义判断筛选方案。
    print("不同方法的选中列可能不同，应结合验证结果与业务含义判断。")


if __name__ == "__main__":
    main()
