# 第6章 模型训练、评估与推理

本章沿着数据集划分、模型选择与训练、评估调优、预测结果剖析、可解释性分析、模型导出与推理的顺序，展示从训练数据到新样本预测的主要环节。示例用于理解书稿中的方法；部分数据由脚本生成，Titanic 与 Boston 房价示例使用仓库中的数据加载函数。

## 子章节与配套代码

| 书稿小节 | 代码文件 | 用途 |
| --- | --- | --- |
| 6.0 模型训练、评估与推理的概述 |  | 认识模型构建、训练、评估、调优、导出和推理的关系。先阅读书稿，建立本章流程。 |
| 6.1 数据集划分 | [6.1、数据集划分/6.1_dataset_splitting.py](6.1、数据集划分/6.1_dataset_splitting.py) | 区分训练集、验证集与测试集，认识随机、分层、交叉验证和时间序列划分。先观察划分方式及每种方式保留的数据顺序和类别比例。 |
| 6.2.1 选择算法 | [6.2、模型选择与训练/6.2.1_model_selection_examples.py](6.2、模型选择与训练/6.2.1_model_selection_examples.py) | 按分类、回归和聚类任务选择模型。对照不同任务的目标变量和训练方式阅读。 |
| 6.2.2 模型训练 | [6.2、模型选择与训练/6.2.2_training_resampling_methods.py](6.2、模型选择与训练/6.2.2_training_resampling_methods.py) | 了解模型拟合及留出、重复分层划分、自助采样的评估方式。重复分层划分与自助采样分别在训练集内部、袋外样本上评估；独立测试集只用于最终一次评价。 |
| 6.3.1 模型评估 | [6.3、模型评估与调优/6.3.1_classification_metrics.py](6.3、模型评估与调优/6.3.1_classification_metrics.py)<br>[6.3、模型评估与调优/6.3.1_regression_metrics.py](6.3、模型评估与调优/6.3.1_regression_metrics.py)<br>[6.3、模型评估与调优/6.3.1_clustering_metrics.py](6.3、模型评估与调优/6.3.1_clustering_metrics.py) | 根据分类、回归和聚类任务选择不同指标。可按分类、回归、聚类的顺序分别阅读；ROC AUC 和平均精确率 AP 使用预测概率，聚类同时比较内部与外部指标。 |
| 6.3.2.1 超参数及其调优简介 |  | 理解模型参数、超参数及调优目标。先阅读概念，再看下一小节的策略比较。 |
| 6.3.2.2 模型调参方法 | [6.3、模型评估与调优/6.3.2_tuning_methods.py](6.3、模型评估与调优/6.3.2_tuning_methods.py)<br>[6.3、模型评估与调优/6.3.2_tuning_visualization.py](6.3、模型评估与调优/6.3.2_tuning_visualization.py) | 比较手动、随机、贪心、网格和贝叶斯调参，并观察搜索结果。先看五种策略的搜索范围与交叉验证，再看 LightGBM 参数网格的热力图和曲面图；贝叶斯策略需单独启用。 |
| 6.4.1 Bad-case 分析 | [6.4、模型预测结果剖析/6.4.1_classification_bad_cases.py](6.4、模型预测结果剖析/6.4.1_classification_bad_cases.py)<br>[6.4、模型预测结果剖析/6.4.1_regression_bad_cases.py](6.4、模型预测结果剖析/6.4.1_regression_bad_cases.py) | 通过误分类样本或较大预测误差定位模型的薄弱场景。分别对照真实值与预测值，观察错误样本及误差分布。 |
| 6.4.2 特征重要性挖掘 | [6.4、模型预测结果剖析/6.4.2_feature_importance.py](6.4、模型预测结果剖析/6.4.2_feature_importance.py) | 从模型自身的参数或内置重要性观察特征作用。T1 看线性模型系数，T2 看树模型内置重要性，T3 看神经网络首层权重。T3 仅作启发式观察。 |
| 6.5.1 模型可解释相关图的简介 |  | 认识力矩图、瀑布图和部分依赖图的表达方式。先阅读书稿图示，再看代码中的解释方法。 |
| 6.5.2 代码实战 | [6.5、模型可解释性分析/6.5.2_model_explainability.py](6.5、模型可解释性分析/6.5.2_model_explainability.py) | 比较不同方法解释整体模型、单个预测和特征局部效应的视角。T1 SHAP、T2 LIME、T3 PDP、T4 ALE；每次通过 `--method` 选择一种方法。LIME 报告写入本小节的 `outputs` 目录。 |
| 6.6.1 模型导出 |  | 认识模型序列化及不同格式的作用。先阅读书稿，再对照 6.6.3 中的保存方式。 |
| 6.6.2 模型推理 |  | 理解对无标签新数据进行预处理、加载和预测。注意训练与推理使用相同的特征处理规则。 |
| 6.6.3 代码实战 | [6.6、模型导出并推理/6.6.3_model_export_inference.py](6.6、模型导出并推理/6.6.3_model_export_inference.py) | 对比 sklearn、PyTorch、TensorFlow 的导出、加载和新样本推理。通过 `--framework` 分别选择三个框架；PyTorch 和 TensorFlow 的标准化器与模型分别保存并重新加载。ONNX 可选导出后由 ONNX Runtime 推理；PMML 仅导出，加载推理需兼容的 PMML 运行时。运行结果写入本小节的 `outputs` 目录。 |

## 建议阅读顺序

依次看数据划分、模型训练、评估与调参，再看预测误差和可解释性分析，最后看模型导出与推理。

## 注意事项

从仓库根目录运行脚本；使用 `shared.dataset` 的示例需让仓库根目录位于 Python 模块搜索路径中。Titanic 和 Boston 房价数据经由 OpenML 获取，首次加载可能需要网络。软件环境参见 [conda-spec-win64.txt](../conda-spec-win64.txt)。该文件是 Windows 平台的 Conda 环境导出清单，在其他系统上应准备兼容环境。

6.5.2 的 SHAP、LIME 可视化与 6.6.3 的 PMML、ONNX 导出依赖相应工具及兼容版本，其中 SHAP、LIME 和 ONNX 相关工具未列在当前 `conda-spec-win64.txt` 中。6.6.3 通过 `pickle` 加载由本脚本创建的 sklearn 模型和标准化器文件；不要加载来源不明的 pickle 文件。
