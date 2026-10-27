# 第5章 数据工程(数据分析+数据处理)

本章沿“清洗 → 分析 → 构造特征 → 特征三化 → 优化特征集 → 导出”的顺序，将书稿中的处理方法对应到小型示例。脚本中的 Step 标出处理流程，T1、T2 等标出同一主题的不同方案；带开关的长脚本可只阅读或运行所需方案。

## 子章节与配套代码

| 书稿小节 | 代码文件 | 用途 |
| --- | --- | --- |
| 5.0 数据工程概述 |  | 认识数据工程的目标与后续处理顺序。先阅读书稿，建立清洗、分析和特征工程的整体框架。 |
| 5.1.1 数据对齐 | [5.1、数据清洗/5.1_data_cleaning.py](5.1%E3%80%81%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97/5.1_data_cleaning.py) | 统一数据类型、日期格式和类别语义。开启 data_alignment，先看对齐前后的字段值。 |
| 5.1.2 缺失值的分析与处理 | [5.1、数据清洗/5.1_data_cleaning.py](5.1%E3%80%81%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97/5.1_data_cleaning.py) | 统计、可视化缺失值，并对比删除与增补。阅读同一脚本的 missing_values_statistics 与 missing_values_handle 两组开关。 |
| 5.1.3 异常值的分析与处理 | [5.1、数据清洗/5.1_data_cleaning.py](5.1%E3%80%81%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97/5.1_data_cleaning.py) | 用统计图、距离与聚类方法检测异常值，再比较删除、截断和变换。按脚本中的异常值开关逐项观察小型 person_info 数据。 |
| 5.1.4 特殊值的分析与处理 | [5.1、数据清洗/5.1_data_cleaning.py](5.1%E3%80%81%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97/5.1_data_cleaning.py)<br>[5.1、数据清洗/5.1.4_sql_default_values.py](5.1%E3%80%81%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97/5.1.4_sql_default_values.py) | 观察特定取值的数量和占比。开启 detect_special_values，对照 Titanic 的 sibsp 与 parch；再看 SQL 示例，比较省略字段、显式写入 0 和 NULL 时的结果。 |
| 5.2.0 概述 |  | 明确分析数据的研究对象和用途。先阅读书稿，再按后续小节看图。 |
| 5.2.1 校验两份数据集是否同分布 | [5.2、数据分析与处理(侧重挖掘信息)/5.2_data_analysis.py](5.2%E3%80%81%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E5%A4%84%E7%90%86%28%E4%BE%A7%E9%87%8D%E6%8C%96%E6%8E%98%E4%BF%A1%E6%81%AF%29/5.2_data_analysis.py) | 比较训练集与测试集的数值和类别特征分布。开启 Distribution_PK_Train_Test，检查划分后的分布。 |
| 5.2.2 目标变量的分析与处理 | [5.2、数据分析与处理(侧重挖掘信息)/5.2_data_analysis.py](5.2%E3%80%81%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E5%A4%84%E7%90%86%28%E4%BE%A7%E9%87%8D%E6%8C%96%E6%8E%98%E4%BF%A1%E6%81%AF%29/5.2_data_analysis.py) | 分别观察回归目标与分类目标的分布。按需开启 num_label_distribution 或 cat_label_distribution。 |
| 5.2.3 类别型特征分析 | [5.2、数据分析与处理(侧重挖掘信息)/5.2_data_analysis.py](5.2%E3%80%81%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E5%A4%84%E7%90%86%28%E4%BE%A7%E9%87%8D%E6%8C%96%E6%8E%98%E4%BF%A1%E6%81%AF%29/5.2_data_analysis.py) | 观察类别分布及其与目标变量的关系。阅读 cat_feature_distribution、cat_feature_relation 和组合图。 |
| 5.2.4 数值型特征分析与处理 | [5.2、数据分析与处理(侧重挖掘信息)/5.2_data_analysis.py](5.2%E3%80%81%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E5%A4%84%E7%90%86%28%E4%BE%A7%E9%87%8D%E6%8C%96%E6%8E%98%E4%BF%A1%E6%81%AF%29/5.2_data_analysis.py) | 查看偏度、峰度、单变量分布、相关性和平行坐标图。依次阅读单列图、关联热图与归一化前后对照。 |
| 5.2.5 组合关联统计分析 | [5.2、数据分析与处理(侧重挖掘信息)/5.2_data_analysis.py](5.2%E3%80%81%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E5%A4%84%E7%90%86%28%E4%BE%A7%E9%87%8D%E6%8C%96%E6%8E%98%E4%BF%A1%E6%81%AF%29/5.2_data_analysis.py) | 把类别与数值特征放在同一图中观察关系。阅读分层直方图、联合图和成对关系图。 |
| 5.3.1 基于常识经验或领域知识构造特征 | [5.3、构造特征/5.3_feature_construction.py](5.3%E3%80%81%E6%9E%84%E9%80%A0%E7%89%B9%E5%BE%81/5.3_feature_construction.py) | 从时间、地理距离、姓名规则和经验聚类得到新列。在脚本中查看对应 T1—T4，按需开启方案。 |
| 5.3.2 基于纯技术构造特征 | [5.3、构造特征/5.3_feature_construction.py](5.3%E3%80%81%E6%9E%84%E9%80%A0%E7%89%B9%E5%BE%81/5.3_feature_construction.py)<br>[5.3、构造特征/5.3.2_graph_features.py](5.3%E3%80%81%E6%9E%84%E9%80%A0%E7%89%B9%E5%BE%81/5.3.2_graph_features.py) | 对数变换、分箱、交互、叶节点、RFM 和图结构各有不同输入。先看小型表格构造，再看 T7 图特征脚本中的 MovieLens 评分关系。 |
| 5.3.3 基于业务规则和意义构造特征 | [5.3、构造特征/5.3_feature_construction.py](5.3%E3%80%81%E6%9E%84%E9%80%A0%E7%89%B9%E5%BE%81/5.3_feature_construction.py) | 用行业与职业的风险等级组合出新业务特征。阅读 design_features_by_Business 示例及其输出。 |
| 5.3.4 利用深度学习技术自动构造特征 | [5.3、构造特征/5.3.4_deep_feature_extraction.py](5.3%E3%80%81%E6%9E%84%E9%80%A0%E7%89%B9%E5%BE%81/5.3.4_deep_feature_extraction.py) | 对照书稿 T1 自动编码器、T2 CNN/RNN 与 T4 预训练模型提取的特征；书稿 T3 GAN 为概念介绍，本示例未实现。先看本地训练的 T1、T2，再按需开启 T4。 |
| 5.3.5 相关库或者框架 |  | 认识可用于特征构造的工具。本节以书稿概念说明为主。 |
| 5.4.0 特征三化概述 |  | 理解归一化、编码化和向量化的关系。先阅读书稿，再比较后续三组示例。 |
| 5.4.1 数值型特征归一化 | [5.4、特征三化/5.4.1_feature_scaling.py](5.4%E3%80%81%E7%89%B9%E5%BE%81%E4%B8%89%E5%8C%96/5.4.1_feature_scaling.py) | 比较 Min-Max 和 Z-Score 对数值分布的影响。对照 Titanic 与 Boston 的图；Boston target 只用于复现图示。 |
| 5.4.2.0 类别型特征编码化概述 |  | 理解结构化类别与文本表示的差异。先阅读书稿，再看两类编码示例。 |
| 5.4.2.1 狭义的类别型特征编码 | [5.4、特征三化/5.4.2.1_structured_categorical_encoding.py](5.4%E3%80%81%E7%89%B9%E5%BE%81%E4%B8%89%E5%8C%96/5.4.2.1_structured_categorical_encoding.py) | 对比独热、序数、标签、目标和哈希编码。依次阅读 T1—T5，注意目标编码只从训练样本计算。 |
| 5.4.2.2 广义的文本型数据编码 | [5.4、特征三化/5.4.2.2_T1_text_representation_comparison.py](5.4%E3%80%81%E7%89%B9%E5%BE%81%E4%B8%89%E5%8C%96/5.4.2.2_T1_text_representation_comparison.py)<br>[5.4、特征三化/5.4.2.2_T2_text_embedding_methods.py](5.4%E3%80%81%E7%89%B9%E5%BE%81%E4%B8%89%E5%8C%96/5.4.2.2_T2_text_embedding_methods.py) | 比较同一个词的词袋、TF-IDF、Word2Vec 表示，以及不同粒度的嵌入。先看 T1 的同词对照，再看 T2 的词、文档和句向量。 |
| 5.4.3 特征向量化 | [5.4、特征三化/5.4.3_feature_vectorization.py](5.4%E3%80%81%E7%89%B9%E5%BE%81%E4%B8%89%E5%8C%96/5.4.3_feature_vectorization.py) | 组合数值填补与缩放、类别填补与独热编码。观察输入列、ColumnTransformer 和最终向量形状。 |
| 5.5.0 优化特征集概述 |  | 区分删除、筛选与降维。先阅读书稿中的三种处理目标。 |
| 5.5.1 特征删除 | [5.5、优化特征集/5.5.1_feature_removal.py](5.5%E3%80%81%E4%BC%98%E5%8C%96%E7%89%B9%E5%BE%81%E9%9B%86/5.5.1_feature_removal.py) | 比较低方差、特征间相关性和随机森林重要性。依次看 T1—T3；指标仅提供删除候选。 |
| 5.5.2 特征筛选 | [5.5、优化特征集/5.5.2_feature_selection.py](5.5%E3%80%81%E4%BC%98%E5%8C%96%E7%89%B9%E5%BE%81%E9%9B%86/5.5.2_feature_selection.py) | 比较过滤式、RFE 包裹式和嵌入式。依次看 T1—T3；先划分数据，仅在训练集拟合填补、编码、缩放与筛选，验证集只用于 RFE 特征数比较。 |
| 5.5.3 特征降维 | [5.5、优化特征集/5.5.3_dimensionality_reduction.py](5.5%E3%80%81%E4%BC%98%E5%8C%96%E7%89%B9%E5%BE%81%E9%9B%86/5.5.3_dimensionality_reduction.py) | 在 Iris 数据上对照 PCA、SVD、MDS、t-SNE 和 LDA。先看三维无监督图，再看 LDA 的二维判别图。 |
| 5.6 特征导出 | [5.6、特征导出/5.6_feature_export.py](5.6%E3%80%81%E7%89%B9%E5%BE%81%E5%AF%BC%E5%87%BA/5.6_feature_export.py) | 把预处理后的 Titanic 表导出为多种常见格式。按脚本顺序观察 CSV、JSON、Excel、Parquet、SQLite 和 HDF5。 |

## 建议阅读顺序

按 5.1 清洗、5.2 分析、5.3 构造、5.4 特征三化、5.5 优化、5.6 导出的顺序阅读；同一小节有多个方案时，对照脚本中的 T 标记。

## 注意事项

从仓库根目录选择需要的脚本运行。使用 `shared.dataset` 的脚本在 macOS/Linux 终端可设置 `PYTHONPATH=.`；Windows 终端需使用相应的环境变量设置方式。环境清单见 [conda-spec-win64.txt](../conda-spec-win64.txt)，这是 Windows Conda 导出清单，不是 pip 的 `-r` 格式。

Titanic、Boston 和部分 Iris 数据来自 OpenML；IMDb 数据和预训练句向量模型首次加载也可能需要网络。5.3.2 图特征示例使用本章附带的 `degree_centrality_clustering_coefficient.csv` 中的原始评分列。5.3 的决策树图片导出需要系统 Graphviz 程序；未安装时仍可观察叶节点编码。

5.3 的演示 CSV、树图及 5.6 的六种导出文件会在运行时写入本章 `output/`，避免覆盖随书数据。书稿中的方法比较旨在帮助理解处理流程，不能把单次评分或图形直接作为真实项目的模型结论。
