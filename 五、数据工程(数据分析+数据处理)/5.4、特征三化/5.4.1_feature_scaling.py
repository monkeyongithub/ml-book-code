# -*- coding: utf-8 -*-
# 文件用途：对比 Min-Max 归一化与 Z-Score 标准化。
# 对应章节/小节：第 5 章 5.4.1 数值型特征归一化
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.4、特征三化/5.4.1_feature_scaling.py"
# 备注：Boston target 只用于复现书稿分布图，不作为模型输入。

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from shared.dataset import boston_df as load_boston
from shared.dataset import titanic_df as load_titanic


def plot_scaling(data, columns, title):
    # 比较同一组数据的原始分布、Min-Max 结果和 Z-Score 结果。
    numeric = data[list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median())

    # T1、Min-Max 归一化把各列映射到 [0, 1]。
    min_max = pd.DataFrame(
        MinMaxScaler().fit_transform(numeric), columns=columns
    )
    # T2、Z-Score 标准化将各列转换为零均值、单位方差。
    z_score = pd.DataFrame(
        StandardScaler().fit_transform(numeric), columns=columns
    )

    # Step3、按书稿的三个特征绘图，观察两种变换对分布的影响。
    fig, axes = plt.subplots(1, len(columns), figsize=(18, 5))
    for axis, column in zip(axes, columns):
        axis.hist(numeric[column], bins=30, alpha=0.5, label="原始值")
        axis.hist(min_max[column], bins=30, alpha=0.5, label="Min-Max")
        axis.hist(z_score[column], bins=30, alpha=0.5, label="Z-Score")
        axis.set_title(f"{title}：{column}")
        axis.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Step1、载入书稿图示采用的 Titanic 和 Boston 数据。
    titanic, _ = load_titanic()
    boston, _ = load_boston()

    # Step2、分别选择书稿中的三个观测列；Boston target 只用于复现分布图。
    plot_scaling(titanic, ("sibsp", "age", "fare"), "Titanic")
    plot_scaling(boston, ("CRIM", "LSTAT", "target"), "Boston")


if __name__ == "__main__":
    main()
