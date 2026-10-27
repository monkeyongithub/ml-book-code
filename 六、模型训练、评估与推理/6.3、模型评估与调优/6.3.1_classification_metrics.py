# -*- coding: utf-8 -*-
# 文件用途：以 Titanic 二分类示例计算混淆矩阵、准确率、精确率、召回率、F1、ROC AUC 和平均精确率 AP。
# 对应章节/小节：第 6 章 6.3.1 模型评估：分类任务
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.3、模型评估与调优/6.3.1_classification_metrics.py
# 备注：预处理仅在训练集拟合；AP 使用预测概率计算。

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from shared.dataset import titanic_df


def main():
    # Step1、读取书中使用的 Titanic 数据，先切分再学习填补与编码规则。
    frame, target_name = titanic_df()
    numeric = ["pclass", "age", "sibsp", "fare"]
    categorical = ["sex", "embarked"]
    features = frame[numeric + categorical].copy()
    labels = LabelEncoder().fit_transform(frame[target_name].astype(str))
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", RandomForestClassifier(
                n_estimators=100, random_state=42
            )),
        ]
    )

    # Step2、只在训练集拟合模型，在独立测试集上预测类别和正类概率。
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]

    # Step3、区分阈值指标与排序指标；AUC 必须使用概率而非硬分类标签。
    print(f"准确率：{accuracy_score(y_test, predicted):.3f}")
    print(f"精确率：{precision_score(y_test, predicted):.3f}")
    print(f"召回率：{recall_score(y_test, predicted):.3f}")
    print(f"F1：{f1_score(y_test, predicted):.3f}")
    print(f"ROC AUC：{roc_auc_score(y_test, probability):.3f}")
    print(f"平均精确率 AP：{average_precision_score(y_test, probability):.3f}")

    # 分开绘制混淆矩阵、ROC 与 PR，避免把不同坐标混在一张图上。
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, ax=axes[0])
    RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1])
    PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[2])
    axes[0].set_title("混淆矩阵")
    axes[1].set_title("ROC 曲线")
    axes[2].set_title("PR 曲线")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
