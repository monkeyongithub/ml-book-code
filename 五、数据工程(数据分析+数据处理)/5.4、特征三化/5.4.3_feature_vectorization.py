# -*- coding: utf-8 -*-
# 文件用途：用 ColumnTransformer 把 Titanic 数值和类别特征转换为模型可用的向量。
# 对应章节/小节：第 5 章 5.4.3 特征向量化
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.4、特征三化/5.4.3_feature_vectorization.py"
# 备注：此处展示特征变换流程，不训练预测模型。

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.dataset import titanic_df


def main():
    # Step1、载入 Titanic 数据，分离标签并明确数值列与类别列。
    frame, label_name = titanic_df()
    numeric_columns = ["age", "fare"]
    category_columns = ["pclass", "sex", "embarked"]
    features = frame[numeric_columns + category_columns].copy()
    labels = frame[label_name]
    print("输入特征形状：", features.shape, "标签形状：", labels.shape)

    # Step2、分别处理缺失值、缩放数值列、独热编码类别列，再组合成向量。
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    vectorizer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("category", category_pipeline, category_columns),
    ])
    vectors = vectorizer.fit_transform(features)

    # Step3、核对输出列名和前五条向量，确认标签没有混入输入。
    preview = vectors[:5]
    if hasattr(preview, "toarray"):
        preview = preview.toarray()
    print("向量化后形状：", vectors.shape)
    print("输出列名：", vectorizer.get_feature_names_out().tolist())
    print("前五条向量：", preview)


if __name__ == "__main__":
    main()
