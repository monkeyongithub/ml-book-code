# -*- coding: utf-8 -*-
# 文件用途：用 TensorBoard 记录 Titanic 分类网络的训练过程。
# 对应章节/小节：第 7 章 7.4.3 模型监控代码实战
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 python 七、模型发布、部署与监控/7.4、模型监控/7.4.3_T2_tensorboard.py
# 备注：验证集用于训练期间观察；独立测试集只用于训练后的最终评估。

from pathlib import Path

import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = ("pclass", "age", "sibsp", "fare")
CATEGORICAL_COLUMNS = ("sex", "embarked")
CODE_ROOT = Path(__file__).resolve().parents[2]
LOG_DIRECTORY = (
    CODE_ROOT
    / "七、模型发布、部署与监控"
    / "7.4、模型监控"
    / "logs"
)


def load_and_split_data():
    # Step1、读取书稿中的 Titanic 数据，并分出独立测试集与验证集。
    dataset = fetch_openml("titanic", version=1, as_frame=True)
    features = dataset.data[
        list(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    ].copy()
    labels = dataset.target.astype("float32")

    train_val_x, test_x, train_val_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_val_x,
        train_val_y,
        test_size=0.2,
        stratify=train_val_y,
        random_state=42,
    )
    return train_x, validation_x, test_x, train_y, validation_y, test_y


def preprocess_data(train_x, validation_x, test_x):
    # Step2、只用训练集拟合缺失值填充、编码与标准化规则。
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_COLUMNS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ]
    )
    train = preprocessor.fit_transform(train_x).astype("float32")
    validation = preprocessor.transform(validation_x).astype("float32")
    test = preprocessor.transform(test_x).astype("float32")
    return train, validation, test


def main():
    # T2、使用 TensorFlow 的 TensorBoard 回调记录训练过程。
    train_x, validation_x, test_x, train_y, validation_y, test_y = (
        load_and_split_data()
    )
    train, validation, test = preprocess_data(
        train_x, validation_x, test_x
    )

    # Step3、构建书稿中的小型 Keras 分类网络并记录训练指标。
    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(train.shape[1],)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(LOG_DIRECTORY), histogram_freq=1
    )
    model.fit(
        train,
        train_y,
        epochs=100,
        batch_size=32,
        validation_data=(validation, validation_y),
        callbacks=[tensorboard],
        verbose=0,
    )

    # Step4、训练结束后才使用测试集计算最终指标。
    probabilities = model.predict(test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print(f"准确率：{accuracy_score(test_y, predictions):.4f}")
    print(f"F1：{f1_score(test_y, predictions):.4f}")
    print(f"AUC：{roc_auc_score(test_y, probabilities):.4f}")
    print(f"TensorBoard 日志：{LOG_DIRECTORY}")


if __name__ == "__main__":
    main()
