# -*- coding: utf-8 -*-
# 文件用途：比较低方差、特征相关性和随机森林重要性三种特征删除依据。
# 对应章节/小节：第 5 章 5.5.1 特征删除
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.5、优化特征集/5.5.1_feature_removal.py"
# 备注：三种方案分别解释分布性、重复性和与预测任务的相关性。

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

from shared.dataset import titanic_df


def show_low_variance(features, threshold=0.3):
    """计算单列方差，并指出低于阈值的特征。"""
    encoded = features.copy()
    for column in encoded.select_dtypes(exclude="number").columns:
        values = encoded[column].astype("object").fillna("未知")
        encoded[column] = LabelEncoder().fit_transform(values.astype(str))
    encoded = encoded.fillna(encoded.median(numeric_only=True))

    variances = encoded.var().sort_values(ascending=False)
    selector = VarianceThreshold(threshold=threshold).fit(encoded)
    removed = encoded.columns[~selector.get_support()].tolist()
    print("各特征方差：", variances)
    print("低于方差阈值的特征：", removed)
    print("类别编码只用于演示；编码数值大小不代表类别顺序。")
    variances.plot.bar(figsize=(10, 5), title="Titanic 特征方差")
    plt.ylabel("方差")
    plt.tight_layout()
    plt.show()


def show_feature_correlation(features):
    """用热力图检查数值特征之间的重复信息。"""
    numeric = features.select_dtypes(include="number")
    correlation = numeric.corr(method="pearson")
    print("数值特征相关系数：", correlation)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Titanic 数值特征的皮尔逊相关系数")
    plt.tight_layout()
    plt.show()


def show_model_importance(frame, label_name):
    """训练小型随机森林，观察与生存预测有关的特征。"""
    columns = ["pclass", "sex", "age", "fare", "embarked"]
    features = frame[columns].copy()
    categorical = features.select_dtypes(exclude="number").columns
    numerical = features.select_dtypes(include="number").columns
    for column in categorical:
        modes = features[column].mode(dropna=True)
        features[column] = features[column].fillna(modes.iloc[0] if not modes.empty else "未知")
    features[numerical] = features[numerical].fillna(features[numerical].median())
    features = pd.get_dummies(features, columns=categorical, dtype=int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, frame[label_name].astype(int))
    importance = pd.Series(model.feature_importances_, index=features.columns)
    importance = importance.sort_values(ascending=False)
    print("随机森林特征重要性：", importance)
    importance.plot.bar(figsize=(9, 5), title="Titanic 随机森林特征重要性")
    plt.ylabel("重要性")
    plt.tight_layout()
    plt.show()


def main():
    # Step1、载入书稿使用的 Titanic 数据，并将标签与输入特征分开。
    frame, label_name = titanic_df()
    features = frame.drop(columns=label_name)

    # Step2、分别观察三种特征删除依据；本例只展示候选，不自动删除特征。
    # T1、方差阈值：找出几乎不变化的列。
    show_low_variance(features)
    # T2、皮尔逊相关系数：检查数值列之间是否携带重复信息。
    show_feature_correlation(features)
    # T3、随机森林重要性：观察特征对当前分类任务的贡献。
    show_model_importance(frame, label_name)

    # Step3、对照三种结果，再结合业务含义决定是否删除某列。
    print("请结合分布、相关性和业务意义判断，不要仅凭单一指标删除特征。")


if __name__ == "__main__":
    main()
