# -*- coding: utf-8 -*-
# 文件用途：以 Titanic 分类结果定位误分类样本并观察其特征分布。
# 对应章节/小节：第 6 章 6.4.1 Bad-case 分析：分类任务
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.4、模型预测结果剖析/6.4.1_classification_bad_cases.py
# 备注：误分类示例仅解释预测结果，不以测试集反向调参。

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from shared.dataset import titanic_df


def main():
    # Step1、读取 Titanic 数据；训练集和测试集先分开。
    frame, target_name = titanic_df()
    numeric = ["pclass", "age", "sibsp", "fare"]
    categorical = ["sex", "embarked"]
    features = frame[numeric + categorical].copy()
    labels = LabelEncoder().fit_transform(frame[target_name].astype(str))
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    preparation = ColumnTransformer(
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
            ("preparation", preparation),
            ("classifier", LGBMClassifier(
                n_estimators=100, random_state=42, verbose=-1
            )),
        ]
    )
    model.fit(x_train, y_train)

    # Step2、用测试集得到预测标签，区分正确与错误样本。
    predicted = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    wrong = predicted != y_test
    print(f"误分类数量：{np.count_nonzero(wrong)} / {len(y_test)}")
    print(f"F1：{f1_score(y_test, predicted):.3f}")
    print(f"ROC AUC：{roc_auc_score(y_test, probability):.3f}")

    # Step3、先看混淆矩阵，再比较错误样本在年龄与票价上的位置。
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, ax=axes[0])
    axes[0].set_title("误分类矩阵")
    axes[1].scatter(
        x_test.loc[~wrong, "age"], x_test.loc[~wrong, "fare"],
        alpha=0.35, label="预测正确",
    )
    axes[1].scatter(
        x_test.loc[wrong, "age"], x_test.loc[wrong, "fare"],
        color="red", alpha=0.75, label="误分类",
    )
    axes[1].set(xlabel="年龄", ylabel="票价", title="错误样本的特征位置")
    axes[1].legend()
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
