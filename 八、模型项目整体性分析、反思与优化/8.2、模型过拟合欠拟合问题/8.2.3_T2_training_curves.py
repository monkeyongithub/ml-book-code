# -*- coding: utf-8 -*-
# 文件用途：绘制神经网络训练与验证的损失、准确率曲线。
# 对应章节/小节：第 8 章 8.2.3 代码实战
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.2、模型过拟合欠拟合问题/8.2.3_T2_training_curves.py"。
# 备注：从 OpenML 获取 Titanic 数据；预处理仅在训练集上拟合，验证集只用于观察曲线。

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras

from shared.dataset import titanic_df


def main():
    # T2、通过训练与验证曲线观察拟合状态。
    # Step1、划分训练与验证数据，先划分再拟合缺失值填充和编码器。
    frame, label_name = titanic_df()
    features = frame[["pclass", "sex", "age", "sibsp", "fare", "embarked"]]
    labels = frame[label_name].astype(int)
    x_train, x_valid, y_train, y_valid = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, ["pclass", "age", "sibsp", "fare"]),
            ("category", categorical, ["sex", "embarked"]),
        ]
    )
    train_array = preprocess.fit_transform(x_train).astype("float32")
    valid_array = preprocess.transform(x_valid).astype("float32")

    # Step2、训练小型分类网络，同时记录训练集与验证集的指标。
    keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            keras.Input(shape=(train_array.shape[1],)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_array,
        y_train.to_numpy(dtype=np.float32),
        validation_data=(valid_array, y_valid.to_numpy(dtype=np.float32)),
        epochs=50,
        batch_size=32,
        verbose=0,
    )

    # Step3、比较两组曲线的间距和走向，观察是否出现过拟合迹象。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(
        axes, ("loss", "accuracy"), ("损失变化", "准确率变化")
    ):
        ax.plot(history.history[metric], label="训练集")
        ax.plot(history.history[f"val_{metric}"], label="验证集")
        ax.set(title=title, xlabel="训练轮次", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
