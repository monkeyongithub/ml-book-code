# -*- coding: utf-8 -*-
# 文件用途：演示三维散点图、三维柱状图和三维折线图各自表达的数据关系。
# 对应章节/小节：第 2 章 2.3.1 三维散点图、柱状图、折线图
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 二、数据可视化技术/2.3、三维图/2.3.1_three_dimensional_plots.py --chart scatter
# 备注：散点图使用 Titanic 数据；柱状图和折线图使用模拟季度销量。

import argparse

import matplotlib.pyplot as plt
import numpy as np

from shared.dataset import titanic_df


def plot_scatter():
    """用年龄、票价和舱位等级构建三维乘客散点图。"""
    data, target_name = titanic_df()
    columns = ["age", "fare", "pclass", target_name]
    data = data[columns].dropna()
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(projection="3d")

    # T1、按生还类别给散点着色，比较三维空间中的分布。
    for label, group in data.groupby(target_name):
        axis.scatter(
            group["age"], group["fare"], group["pclass"],
            alpha=0.45, label=f"生还类别 {label}",
        )
    axis.set(xlabel="年龄", ylabel="票价", zlabel="舱位等级")
    axis.set_title("Titanic 乘客三维散点图")
    axis.legend()


def sample_sales():
    """生成用于解释三维柱状图和折线图的示意销量。"""
    years = np.arange(2019, 2023)
    quarters = np.arange(1, 5)
    sales = np.array(
        [[62, 77, 83, 91], [71, 81, 94, 100],
         [80, 92, 107, 116], [89, 101, 120, 132]]
    )
    return years, quarters, sales


def plot_bar():
    years, quarters, sales = sample_sales()
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(projection="3d")

    # T2、每根柱子对应一个年份和季度，柱高表示销量。
    year_grid, quarter_grid = np.meshgrid(years, quarters, indexing="ij")
    axis.bar3d(
        year_grid.ravel(), quarter_grid.ravel(), np.zeros(sales.size),
        0.6, 0.6, sales.ravel(), shade=True,
    )
    axis.set(xlabel="年份", ylabel="季度", zlabel="销量")
    axis.set_xticks(years)
    axis.set_yticks(quarters)
    axis.set_title("季度销量三维柱状图（示意数据）")


def plot_line():
    years, quarters, sales = sample_sales()
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(projection="3d")

    # T3、每条线表示一个年份内四个季度的销量走势。
    for year, values in zip(years, sales):
        axis.plot(quarters, values, zs=year, zdir="x", marker="o", label=str(year))
    axis.set(xlabel="年份", ylabel="季度", zlabel="销量")
    axis.set_xticks(years)
    axis.set_yticks(quarters)
    axis.set_title("季度销量三维折线图（示意数据）")
    axis.legend(title="年份")


def main():
    parser = argparse.ArgumentParser(description="选择一种书稿 2.3.1 的三维图。")
    parser.add_argument(
        "--chart", choices=("scatter", "bar", "line"), default="scatter"
    )
    chart = parser.parse_args().chart

    # Step1、按图型选择乘客数据或模拟销量数据。
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    # Step2、在三个坐标轴中放入与图形含义一致的变量。
    {"scatter": plot_scatter, "bar": plot_bar, "line": plot_line}[chart]()

    # Step3、旋转图形，观察第三个维度如何改变数据解读。
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
