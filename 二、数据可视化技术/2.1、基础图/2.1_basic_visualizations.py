# -*- coding: utf-8 -*-
# 文件用途：按需演示书稿 2.1 节的基础图、多变量图，以及雷达图、词云图和两类树图。
# 对应章节/小节：第 2 章 2.1.1—2.1.3 基础图的简介及代码实现
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 二、数据可视化技术/2.1、基础图/2.1_basic_visualizations.py --chart histogram
# 备注：使用 --chart 选择一种图；treemap 和 decision_tree 使用示意数据，词云图读取本节遮罩图片。

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from shared.dataset import titanic_df

MASK_PATH = Path("二、数据可视化技术/2.1、基础图/images/wordcloud08.jpg")


def prepare_data():
    """载入 Titanic 数据，保留各图所需的列。"""
    data, target_name = titanic_df()
    return data.copy(), target_name


def plot_histogram(data, _):
    # 2.1.1：直方图展示数值特征的分布。
    sns.histplot(data=data, x="age", bins=20, kde=True)
    plt.title("乘客年龄分布")


def plot_bar(data, _):
    # 2.1.1：条形图比较各舱位等级的乘客数量。
    sns.countplot(data=data, x="pclass")
    plt.title("各舱位等级人数")


def plot_line(_, __):
    # 2.1.1：用示意时间序列突出折线图的时间变化含义。
    months = np.arange(1, 7)
    sales = [22, 27, 25, 31, 36, 40]
    plt.plot(months, sales, marker="o")
    plt.xticks(months)
    plt.xlabel("月份")
    plt.ylabel("销量")
    plt.title("月度销量趋势（示意数据）")


def plot_box(data, _):
    # 2.1.1：箱线图比较不同舱位的票价分布及离群点。
    sns.boxplot(data=data, x="pclass", y="fare")
    plt.title("不同舱位的票价分布")


def plot_pie(data, _):
    # 2.1.1：饼状图显示各舱位占总体的比例。
    counts = data["pclass"].value_counts().sort_index()
    plt.pie(counts, labels=counts.index.astype(str), autopct="%.1f%%")
    plt.title("各舱位乘客比例")


def plot_parallel(data, target_name):
    # 2.1.1：抽取少量样本，避免平行坐标图因曲线过多难以阅读。
    columns = ["pclass", "age", "fare"]
    subset = data[columns + [target_name]].dropna()
    subset = subset.sample(n=min(80, len(subset)), random_state=42).copy()
    subset[columns] = MinMaxScaler().fit_transform(subset[columns])
    subset[target_name] = subset[target_name].astype(str)
    pd.plotting.parallel_coordinates(subset, class_column=target_name)
    plt.title("乘客特征的平行坐标图")


def plot_kde(data, _):
    # 2.1.1：核密度曲线更平滑地展示年龄分布。
    sns.kdeplot(data=data, x="age", fill=True)
    plt.title("乘客年龄核密度估计")


def plot_violin(data, _):
    # 2.1.1：小提琴图显示不同舱位的年龄分布形状。
    sns.violinplot(data=data, x="pclass", y="age")
    plt.title("不同舱位的年龄分布")


def plot_scatter(data, target_name):
    # 2.1.2：散点图观察年龄与票价的关系，颜色表示结果类别。
    sns.scatterplot(data=data, x="age", y="fare", hue=target_name, alpha=0.6)
    plt.title("年龄与票价的关系")


def plot_pearson_heatmap(data, _):
    # 2.1.2：Pearson 热力图仅针对数值型字段。
    numeric = data.select_dtypes(include="number")
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("数值特征 Pearson 相关系数")


def plot_mic_heatmap(data, _):
    # 2.1.2：MIC 能补充观察非线性相关，作为热力图的第二种度量。
    from minepy import MINE

    numeric = data[["pclass", "age", "sibsp", "parch", "fare"]].dropna()
    columns = numeric.columns.tolist()
    scores = np.eye(len(columns))
    mine = MINE(alpha=0.6, c=15)
    for i, first in enumerate(columns):
        for j in range(i + 1, len(columns)):
            mine.compute_score(numeric[first], numeric[columns[j]])
            scores[i, j] = scores[j, i] = mine.mic()
    sns.heatmap(
        pd.DataFrame(scores, index=columns, columns=columns),
        annot=True, fmt=".2f", cmap="PiYG", vmin=0, vmax=1,
    )
    plt.title("数值特征最大信息系数 MIC")


def plot_grouped_bar(data, target_name):
    # 2.1.2：按类别比较年龄均值，并用颜色区分生还结果。
    sns.barplot(data=data, x="sex", y="age", hue=target_name)
    plt.title("不同性别与生还结果的平均年龄")


def plot_point(data, target_name):
    # 2.1.2：连接各组估计值，突出组间变化趋势。
    sns.pointplot(data=data, x="pclass", y="age", hue=target_name)
    plt.title("不同舱位的平均年龄变化")


def plot_radar(data, target_name):
    # 2.1.3：将多个数值维度归一化，比较生还类别的平均画像。
    columns = ["pclass", "age", "sibsp", "parch", "fare"]
    subset = data[columns + [target_name]].copy()
    subset[columns] = subset[columns].fillna(subset[columns].median())
    subset[columns] = MinMaxScaler().fit_transform(subset[columns])
    averages = subset.groupby(target_name)[columns].mean()
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    axis = plt.subplot(111, projection="polar")
    for label, values in averages.iterrows():
        closed_values = np.r_[values.to_numpy(), values.iloc[0]]
        axis.plot(closed_angles, closed_values, label=str(label))
        axis.fill(closed_angles, closed_values, alpha=0.1)
    axis.set_xticks(angles, columns)
    axis.set_ylim(0, 1)
    axis.legend(title="是否生还")
    axis.set_title("乘客特征雷达图")


def plot_wordcloud(_, __, font_path=None):
    # 2.1.3：使用随书遮罩图形成书稿中的心形词云。
    from PIL import Image
    from wordcloud import WordCloud

    if not MASK_PATH.is_file():
        raise FileNotFoundError(f"缺少词云遮罩图片：{MASK_PATH}")
    terms = "人工智能 数据分析 机器学习 深度学习 机器人 智能推荐 数据科学"
    mask = np.asarray(Image.open(MASK_PATH).convert("L"))
    cloud = WordCloud(
        font_path=font_path, mask=mask, background_color="white",
        random_state=42,
    ).generate(terms)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("人工智能主题词云")


# T1、矩形树图（Tree Map）：用嵌套矩形的面积表示层级与数量比例。
def plot_treemap(_, __):
    # Step1、构造“部门—品类”两级示意销量，明确每个叶子区域的值。
    groups = {
        "图书": [("小说", 40, "#70A9D6"), ("科普", 20, "#B7D6EB")],
        "文具": [("笔", 25, "#EE9B55"), ("本", 15, "#F5C697")],
    }
    total = sum(value for items in groups.values() for _, value, _ in items)
    _, axis = plt.subplots(figsize=(9, 5))

    # Step2、先按部门总量分配宽度，再按部门内部品类比例分配高度。
    # 每个矩形的面积占比因此等于该品类销量占总体销量的比例。
    left = 0.0
    for group, items in groups.items():
        group_total = sum(value for _, value, _ in items)
        width = group_total / total
        bottom = 0.0
        for category, value, color in items:
            height = value / group_total
            axis.add_patch(Rectangle(
                (left, bottom), width, height,
                facecolor=color, edgecolor="white", linewidth=2,
            ))
            axis.text(
                left + width / 2, bottom + height / 2,
                f"{group} / {category}\n{value}（{value / total:.0%}）",
                ha="center", va="center",
            )
            bottom += height
        axis.add_patch(Rectangle(
            (left, 0), width, 1, fill=False, edgecolor="#333333", linewidth=2,
        ))
        left += width

    # Step3、隐藏坐标轴，直接对照面积、层级标签与总体占比。
    axis.set(xlim=(0, 1), ylim=(0, 1), title="矩形树图（Tree Map，示意销量）")
    axis.axis("off")


# T2、决策树结构图（Tree Diagram）：用节点、分支和叶节点表示逐步判断。
def plot_decision_tree(_, __):
    # Step1、构造两项特征与购买结果的微型示意数据。
    features = np.array([
        [22, 2], [25, 8], [34, 2], [36, 7], [45, 3], [48, 9],
    ])
    labels = np.array([0, 0, 0, 1, 0, 1])

    # Step2、限制树的深度，使节点和分支便于对照书稿中的概念。
    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(features, labels)

    # Step3、绘出判断条件与叶节点结果；它不以矩形面积表达比例。
    _, axis = plt.subplots(figsize=(11, 6))
    plot_tree(
        model, ax=axis, feature_names=["年龄", "近一周访问次数"],
        class_names=["未购买", "购买"], filled=True, rounded=True,
        impurity=False,
    )
    axis.set_title("决策树结构图（Tree Diagram，示意数据）")


CHARTS = {
    "histogram": plot_histogram,
    "bar": plot_bar,
    "line": plot_line,
    "box": plot_box,
    "pie": plot_pie,
    "parallel": plot_parallel,
    "kde": plot_kde,
    "violin": plot_violin,
    "scatter": plot_scatter,
    "heatmap_pearson": plot_pearson_heatmap,
    "heatmap_mic": plot_mic_heatmap,
    "grouped_bar": plot_grouped_bar,
    "point": plot_point,
    "radar": plot_radar,
    "wordcloud": plot_wordcloud,
    "treemap": plot_treemap,
    "decision_tree": plot_decision_tree,
}


def main():
    parser = argparse.ArgumentParser(
        description="选择一类图，观察它表达的数据关系。",
        epilog="2.1.3 的两类树图：treemap 为矩形树图；decision_tree 为决策树结构图。",
    )
    parser.add_argument("--chart", choices=CHARTS, default="histogram")
    parser.add_argument("--font", help="词云使用的中文字体文件路径")
    args = parser.parse_args()
    if args.chart == "wordcloud" and not args.font:
        parser.error("词云包含中文词语，请用 --font 指定可用的中文字体文件。")

    # Step1、确定目标图型；时间趋势、词云和两类树图使用示意数据。
    if args.chart in {"line", "wordcloud", "treemap", "decision_tree"}:
        data, target_name = None, None
    else:
        data, target_name = prepare_data()

    # Step2、调用对应小节的绘图函数，保持每次只显示一种图。
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    if args.chart == "wordcloud":
        plot_wordcloud(data, target_name, font_path=args.font)
    else:
        CHARTS[args.chart](data, target_name)

    # Step3、观察图中变量、分布或关系，再与书稿的图表解读对照。
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
