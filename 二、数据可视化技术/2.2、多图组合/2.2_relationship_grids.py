# -*- coding: utf-8 -*-
# 文件用途：演示双变量关系图、成对关系矩阵和按类别分组的关系网格。
# 对应章节/小节：第 2 章 2.2.1—2.2.3 多图组合的简介及代码实现
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 二、数据可视化技术/2.2、多图组合/2.2_relationship_grids.py --chart joint
# 备注：使用 --chart 分别查看 Jointplot、Pairplot 与 FacetGrid。

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from shared.dataset import titanic_df


def plot_joint(data, kind):
    """在中心关系图周围展示两个变量各自的分布。"""
    # 2.2.1：同一组年龄与舱位等级数据可切换散点、六边形和密度方法。
    subset = data[["age", "pclass"]].copy()
    subset["age"] = pd.to_numeric(subset["age"], errors="coerce")
    subset["pclass"] = pd.to_numeric(subset["pclass"], errors="coerce")
    chart = sns.jointplot(
        data=subset.dropna(),
        x="age", y="pclass", kind=kind, height=7,
    )
    chart.fig.suptitle(f"年龄与舱位等级的联合分布（{kind}）", y=1.02)


def plot_pair(data, target_name):
    """在矩阵中同时查看各数值特征的成对关系与单变量分布。"""
    # 2.2.2：对角线展示分布，其余网格展示两两关系。
    columns = ["age", "fare", "pclass", target_name]
    chart = sns.pairplot(
        data[columns].dropna(), hue=target_name,
        vars=columns[:-1], diag_kind="hist", corner=True,
    )
    chart.fig.suptitle("乘客特征成对关系矩阵", y=1.02)


def plot_facet(data, target_name):
    """分面比较不同性别与生还结果下的年龄和票价关系。"""
    # 2.2.3：每个面板对应一个分组，便于读者比较组间差异。
    chart = sns.FacetGrid(
        data.dropna(subset=["age", "fare", "sex", target_name]),
        col="sex", hue=target_name, height=4,
    )
    chart.map_dataframe(sns.scatterplot, x="age", y="fare", alpha=0.6)
    chart.add_legend(title="是否生还")
    chart.fig.suptitle("按性别分组的年龄与票价关系", y=1.06)


def main():
    parser = argparse.ArgumentParser(description="选择书稿 2.2 节的一种关系图。")
    parser.add_argument(
        "--chart", choices=("joint", "pair", "facet"), default="joint"
    )
    parser.add_argument(
        "--kind", choices=("scatter", "hex", "kde"), default="scatter",
        help="Jointplot 的中心图类型",
    )
    args = parser.parse_args()

    # Step1、载入书稿中的 Titanic 样本，确定目标变量。
    data, target_name = titanic_df()

    # Step2、按 2.2.1、2.2.2 或 2.2.3 的问题选择图形。
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    if args.chart == "joint":
        plot_joint(data, args.kind)
    elif args.chart == "pair":
        plot_pair(data, target_name)
    else:
        plot_facet(data, target_name)

    # Step3、结合边缘分布、矩阵单元或分组面板解读变量关系。
    plt.show()


if __name__ == "__main__":
    main()
