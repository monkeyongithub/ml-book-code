# -*- coding: utf-8 -*-
# 文件用途：训练 Titanic 分类模型，并通过 Flask 提供单条与批量预测接口。
# 对应章节/小节：第 7 章 7.3.4 模型部署代码实战
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 python 七、模型发布、部署与监控/7.3、模型部署/7.3.4_prediction_api.py
# 备注：仅供本机教学演示；两个接口都接收相同的数值化特征字段。

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

NUMERIC_COLUMNS = ("pclass", "age", "sibsp", "fare")
CATEGORICAL_COLUMNS = ("sex", "embarked")
FEATURE_COLUMNS = (
    "pclass",
    "age",
    "sibsp",
    "fare",
    "sex_female",
    "sex_male",
    "embarked_C",
    "embarked_Q",
    "embarked_S",
)


def encode_features(raw_features, numeric_imputer, category_imputer, encoder):
    """按固定列顺序，把原始 Titanic 特征转换为模型输入。"""
    numeric_values = numeric_imputer.transform(raw_features[list(NUMERIC_COLUMNS)])
    categories = category_imputer.transform(
        raw_features[list(CATEGORICAL_COLUMNS)]
    )
    encoded_categories = encoder.transform(categories)
    values = np.column_stack((numeric_values, encoded_categories))
    return pd.DataFrame(
        values, columns=FEATURE_COLUMNS, index=raw_features.index
    )


def train_model():
    # Step1、取得书稿使用的 Titanic 数据，并先划分训练集和测试集。
    dataset = fetch_openml("titanic", version=1, as_frame=True)
    raw_features = dataset.data[
        list(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    ].copy()
    labels = dataset.target.astype(int)
    train_raw, test_raw, y_train, y_test = train_test_split(
        raw_features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    # Step2、仅在训练集拟合填充与编码规则，再以同一规则处理测试集。
    numeric_imputer = SimpleImputer(strategy="median")
    category_imputer = SimpleImputer(strategy="most_frequent")
    numeric_imputer.fit(train_raw[list(NUMERIC_COLUMNS)])
    train_categories = category_imputer.fit_transform(
        train_raw[list(CATEGORICAL_COLUMNS)]
    )
    encoder = OneHotEncoder(
        categories=(("female", "male"), ("C", "Q", "S")),
        handle_unknown="ignore",
        sparse_output=False,
    )
    encoder.fit(train_categories)
    x_train = encode_features(
        train_raw, numeric_imputer, category_imputer, encoder
    )
    x_test = encode_features(test_raw, numeric_imputer, category_imputer, encoder)

    # Step3、训练 LightGBM，并在未参与预处理拟合的测试集上评估。
    model = LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(f"准确率：{accuracy_score(y_test, predictions):.4f}")
    print(f"F1：{f1_score(y_test, predictions):.4f}")
    print(f"AUC：{roc_auc_score(y_test, probabilities):.4f}")
    return model


def records_to_frame(payload, batch=False):
    """校验请求字段，并按训练时的列顺序建立特征表。"""
    records = payload if batch else [payload]
    if not isinstance(records, list) or not records:
        raise ValueError("请求必须包含至少一条样本。")

    rows = []
    expected = set(FEATURE_COLUMNS)
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 条样本必须是 JSON 对象。")
        if set(record) != expected:
            raise ValueError(
                f"第 {index} 条样本字段应为：{', '.join(FEATURE_COLUMNS)}"
            )
        try:
            row = [float(record[column]) for column in FEATURE_COLUMNS]
        except (TypeError, ValueError) as error:
            raise ValueError(f"第 {index} 条样本含非数值字段。") from error
        if not np.isfinite(row).all():
            raise ValueError(f"第 {index} 条样本含非有限数值。")
        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def create_app(model):
    app = Flask(__name__)

    # Step4、单条与批量接口复用同一字段校验和特征列顺序。
    @app.post("/predict")
    def predict_once():
        try:
            features = records_to_frame(request.get_json(silent=True))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        prediction = int(model.predict(features)[0])
        return jsonify({"prediction": prediction})

    @app.post("/predict_batch")
    def predict_batch():
        try:
            features = records_to_frame(request.get_json(silent=True), batch=True)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        predictions = [int(value) for value in model.predict(features)]
        return jsonify({"predictions": predictions})

    return app


def main():
    model = train_model()
    app = create_app(model)
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
