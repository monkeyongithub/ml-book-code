# 第2章 数据可视化技术

本章从基础图、多图组合、三维图到动态图，说明图形如何表达变量分布、变量关系与随时间变化的过程。各脚本是按书稿小节组织的教学示例，可通过命令参数选择要观察的图。

## 子章节与配套代码

| 书稿小节 | 代码文件 | 用途 |
| --- | --- | --- |
| 2.1.1 单维度及其交互可视化 | [2.1、基础图/2.1_basic_visualizations.py](2.1、基础图/2.1_basic_visualizations.py) | 认识直方图、条形图、折线图、箱线图、平行坐标图和饼状图等。从 `--chart histogram` 开始，逐一观察图型所表达的分布或趋势。 |
| 2.1.2 多维度可视化 | [2.1、基础图/2.1_basic_visualizations.py](2.1、基础图/2.1_basic_visualizations.py) | 用散点图、热力图、分组条形图和点图观察变量关系。选择 `scatter`、`heatmap_pearson`、`heatmap_mic`、`grouped_bar` 或 `point`；MIC 图额外使用 `minepy`。 |
| 2.1.3 其它图的简介 | [2.1、基础图/2.1_basic_visualizations.py](2.1、基础图/2.1_basic_visualizations.py) | 了解雷达图、词云图，以及 T1 矩形树图和 T2 决策树结构图。分别用 `--chart radar`、`--chart wordcloud`、`--chart treemap`、`--chart decision_tree` 选择；词云使用本节的遮罩图片，并通过 `--font` 指定中文字体文件。 |
| 2.2.1 单关系图 | [2.2、多图组合/2.2_relationship_grids.py](2.2、多图组合/2.2_relationship_grids.py) | 同时查看两个变量的关系及各自分布。使用 `--chart joint`，并通过 `--kind` 切换散点、六边形或密度形式。 |
| 2.2.2 多变量关系矩阵图 | [2.2、多图组合/2.2_relationship_grids.py](2.2、多图组合/2.2_relationship_grids.py) | 在矩阵中对照多个特征的成对关系。使用 `--chart pair` 观察对角线和非对角线图形。 |
| 2.2.3 数据分组矩阵图 | [2.2、多图组合/2.2_relationship_grids.py](2.2、多图组合/2.2_relationship_grids.py) | 比较不同分组内的特征关系。使用 `--chart facet`，对照不同面板。 |
| 2.3.1 三维散点图、柱状图、折线图 | [2.3、三维图/2.3.1_three_dimensional_plots.py](2.3、三维图/2.3.1_three_dimensional_plots.py) | 观察第三个维度如何进入散点、柱形和趋势图。依次选择 `scatter`、`bar`、`line`；后两者使用示意销量。 |
| 2.3.2 三维标签图—八象空间三维图 | [2.3、三维图/2.3.2_octant_labels.py](2.3、三维图/2.3.2_octant_labels.py) | 用三维正负组合说明八类客户画像的位置。观察坐标、点位与标签的对应关系；数据为示意值。 |
| 2.4.1 动态趋势图 | [2.4、动态图/2.4_dynamic_visualizations.py](2.4、动态图/2.4_dynamic_visualizations.py) | 观察损失值随训练轮次衰减。使用 `--chart trend`，逐帧观察曲线延伸。 |
| 2.4.2 动态轨迹图 | [2.4、动态图/2.4_dynamic_visualizations.py](2.4、动态图/2.4_dynamic_visualizations.py) | 观察点位随时间移动形成的路径。使用 `--chart trajectory`；抽象点位仅用于说明轨迹，不代表真实地理位置。 |
| 2.5 常用的图可视化相关库 |  | 认识常用绘图库及其适用场景。阅读书稿的库比较，再回看各脚本使用的图型。 |

## 建议阅读顺序

先看 2.1 的基础图，再看 2.2 的多图组合；随后比较 2.3 的三维视角与 2.4 的动态变化。

## 注意事项

从仓库根目录运行脚本，环境信息参见 [conda-spec-win64.txt](../conda-spec-win64.txt)。使用 Titanic 示例时，首次从 OpenML 获取数据可能需要网络。词云图需要本节 `images/wordcloud08.jpg` 遮罩图片；中文字体通过 `--font` 传入。动态图在图形窗口中播放，无需预先生成 HTML 或 GIF。
