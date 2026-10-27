# -*- coding: utf-8 -*-
# 文件用途：对比线性模型、树模型和神经网络首层权重的特征解释视角。
# 对应章节/小节：第 6 章 6.4.2 特征重要性挖掘
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.4、模型预测结果剖析/6.4.2_feature_importance.py
# 备注：神经网络首层权重仅为粗略观察，不等于可靠的全局特征重要性。

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from shared.dataset import titanic_df


def load_features():
    # Step1、先切分 Titanic 数据，再只用训练集拟合填补、编码和标准化。
    frame, target_name = titanic_df()
    numeric = ["pclass", "age", "sibsp", "fare"]
    categorical = ["sex", "embarked"]
    features = frame[numeric + categorical].copy()
    labels = LabelEncoder().fit_transform(frame[target_name].astype(str))
    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
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
        ],
        sparse_threshold=0.0,
    )
    transformed = preprocessing.fit_transform(x_train)
    names = preprocessing.get_feature_names_out()
    return transformed, y_train, names


def show_linear_models(features, labels, names):
    # T1、线性模型：比较标准化后的回归系数与 Lasso 系数路径。
    models = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (title, model) in zip(axes.flat[:3], models.items()):
        model.fit(features, labels)
        axis.barh(names, model.coef_)
        axis.set_title(title)
        axis.set_xlabel("系数；正负号表示方向")

    alphas = np.logspace(-3, 0, 12)
    coefficient_path = [
        Lasso(alpha=alpha, max_iter=10000).fit(features, labels).coef_
        for alpha in alphas
    ]
    axes[1, 1].plot(alphas, coefficient_path)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(xlabel="Lasso alpha", ylabel="系数", title="Lasso 系数路径")
    figure.tight_layout()
    plt.show()


def show_tree_models(features, labels, names):
    # T2、树模型：各模型的内置重要性口径不同，先各自归一化后再并排看。
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    models = {
        "随机森林": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, eval_metric="logloss", random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1
        ),
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (title, model) in zip(axes, models.items()):
        model.fit(features, labels)
        importance = model.feature_importances_.astype(float)
        total = importance.sum()
        if total:
            importance /= total
        axis.barh(names, importance)
        axis.set(xlabel="模型内归一化重要性", title=title)
    figure.tight_layout()
    plt.show()


def show_neural_weights(features, labels, names):
    # T3、神经网络：观察首层权重的平均绝对值，仅作参数层面的启发。
    import torch
    from torch import nn

    torch.manual_seed(42)
    inputs = torch.as_tensor(features, dtype=torch.float32)
    targets = torch.as_tensor(labels.reshape(-1, 1), dtype=torch.float32)
    model = nn.Sequential(nn.Linear(inputs.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.BCEWithLogitsLoss()
    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
    weights = model[0].weight.detach().abs().mean(dim=0).cpu().numpy()
    plt.barh(names, weights)
    plt.xlabel("首层权重平均绝对值")
    plt.title("神经网络首层权重观察")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="比较三类模型的特征解释方式")
    parser.add_argument(
        "--method", choices=["linear", "tree", "neural"], default="linear"
    )
    method = parser.parse_args().method
    features, labels, names = load_features()
    # Step2、按书稿的 T1/T2/T3 选择一种模型视角。
    {
        "linear": show_linear_models,
        "tree": show_tree_models,
        "neural": show_neural_weights,
    }[method](features, labels, names)


if __name__ == "__main__":
    main()
