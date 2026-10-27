# -*- coding: utf-8 -*-
# 文件用途：比较 Platt、Isotonic 及两级概率校准的效果。
# 对应章节/小节：第 8 章 8.4.2 单算法优化
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.4、算法层面优化/8.4.2_probability_calibration.py"。
# 备注：先划分训练、二级校准和测试集；测试标签仅用于最终评价。

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.dataset import titanic_df


def make_classifier():
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, ["pclass", "age", "sibsp", "fare"]),
            ("category", categorical, ["sex", "embarked"]),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def report(name, labels, probabilities):
    print(
        f"{name}：AUC={roc_auc_score(labels, probabilities):.3f}，"
        f"Brier={brier_score_loss(labels, probabilities):.3f}"
    )


def main():
    # Step1、先划分数据，测试集不参与预处理、模型拟合或概率校准。
    frame, label_name = titanic_df()
    features = frame[["pclass", "sex", "age", "sibsp", "fare", "embarked"]]
    labels = frame[label_name].astype(int)
    x_dev, x_test, y_dev, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=42
    )

    # T1、Platt 校准：在训练集内部交叉验证后拟合 sigmoid。
    platt = CalibratedClassifierCV(
        estimator=make_classifier(), method="sigmoid", cv=5
    )
    platt.fit(x_train, y_train)
    platt_prob = platt.predict_proba(x_test)[:, 1]

    # T2、Isotonic 校准：使用相同的训练数据和交叉验证设置。
    isotonic = CalibratedClassifierCV(
        estimator=make_classifier(), method="isotonic", cv=5
    )
    isotonic.fit(x_train, y_train)
    isotonic_prob = isotonic.predict_proba(x_test)[:, 1]

    # T3、组合校准：第二级只在预留校准集上学习，不使用测试标签。
    second_stage = IsotonicRegression(out_of_bounds="clip")
    second_stage.fit(platt.predict_proba(x_cal)[:, 1], y_cal)
    combined_prob = second_stage.predict(platt_prob)

    # Step2、在同一测试集上评价概率质量并绘制可靠性曲线。
    results = {
        "Platt": platt_prob,
        "Isotonic": isotonic_prob,
        "Platt + Isotonic": combined_prob,
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (name, probabilities) in zip(axes, results.items()):
        report(name, y_test, probabilities)
        observed, predicted = calibration_curve(
            y_test, probabilities, n_bins=8, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="理想校准")
        ax.set(title=name, xlabel="平均预测概率")
        ax.legend()
    axes[0].set_ylabel("实际正例比例")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
