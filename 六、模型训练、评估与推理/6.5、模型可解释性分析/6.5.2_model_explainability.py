# -*- coding: utf-8 -*-
# 文件用途：对比 SHAP、LIME、PDP 与 ALE 的模型解释视角。
# 对应章节/小节：第 6 章 6.5.2 模型可解释性代码实战
# 作者：NYY
# 创建日期：2024-08-14
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.5、模型可解释性分析/6.5.2_model_explainability.py --method shap
# 备注：可将 --method 换为 lime、pdp 或 ale；LIME 报告保存在本小节的 outputs 目录。

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from shared.dataset import titanic_df


def prepare_data():
    # Step1、先划分 Titanic 数据，再从训练集学习缺失值填补规则。
    frame, target_name = titanic_df()
    numeric = ["pclass", "age", "sibsp", "fare"]
    categorical = ["sex", "embarked"]
    features = frame[numeric + categorical].copy()
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame[target_name].astype(str))
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    medians = x_train[numeric].median()
    modes = x_train[categorical].mode().iloc[0]
    for sample in (x_train, x_test):
        sample[numeric] = sample[numeric].fillna(medians)
        sample[categorical] = sample[categorical].fillna(modes)
    train_encoded = pd.get_dummies(x_train).astype(float)
    test_encoded = pd.get_dummies(x_test).reindex(
        columns=train_encoded.columns, fill_value=0
    ).astype(float)
    return train_encoded, test_encoded, y_train, y_test, encoder.classes_


def show_shap(model, train_features, test_features):
    # T1、SHAP：对同一模型展示总体贡献和单个样本的贡献。
    import shap

    explainer = shap.TreeExplainer(model)
    explanation = explainer(test_features.iloc[:50])
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    shap.plots.beeswarm(explanation, show=False)
    plt.title("SHAP 总体贡献")
    plt.show()
    shap.plots.waterfall(explanation[0], show=False)
    plt.title("SHAP 单个样本贡献")
    plt.show()


def show_lime(model, train_features, test_features, class_names):
    # T2、LIME：围绕一条预测样本拟合局部可解释近似。
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        train_features.to_numpy(),
        feature_names=train_features.columns.tolist(),
        class_names=class_names.tolist(),
        mode="classification",
        random_state=42,
    )
    explanation = explainer.explain_instance(
        test_features.iloc[0].to_numpy(),
        model.predict_proba,
        num_features=min(8, test_features.shape[1]),
    )
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lime_example.html"
    explanation.save_to_file(str(output_path))
    print(f"LIME 局部解释已保存：{output_path}")
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()


def show_pdp(model, train_features):
    # T3、PDP：对年龄及年龄和票价的组合取边际平均预测。
    PartialDependenceDisplay.from_estimator(
        model, train_features,
        features=["age", ("age", "fare")],
        grid_resolution=20,
    )
    plt.suptitle("特征的部分依赖关系")
    plt.tight_layout()
    plt.show()


def first_order_ale(model, frame, feature, bins=10):
    """通过相邻区间的预测差计算一维 ALE 教学示例。"""
    values = frame[feature].to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        raise ValueError("该特征没有足够的不同取值，无法绘制 ALE。")
    local_effects = []
    counts = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        # 最后一个区间包含右端点，避免遗漏最大值。
        mask = (values >= lower) & (
            (values <= upper) if index == len(edges) - 2 else (values < upper)
        )
        subset = frame.loc[mask]
        counts.append(len(subset))
        if subset.empty:
            local_effects.append(0.0)
            continue
        left, right = subset.copy(), subset.copy()
        left[feature], right[feature] = lower, upper
        difference = (
            model.predict_proba(right)[:, 1]
            - model.predict_proba(left)[:, 1]
        )
        local_effects.append(float(difference.mean()))
    effects = np.cumsum(local_effects)
    effects -= np.average(effects, weights=np.maximum(counts, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, effects


def show_ale(model, train_features):
    # T4、ALE：积累局部预测差，观察年龄的平均局部效应。
    centers, effects = first_order_ale(model, train_features, "age")
    plt.plot(centers, effects, marker="o")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("年龄")
    plt.ylabel("中心化局部效应")
    plt.title("年龄的一维 ALE")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="对比四种模型解释方法")
    parser.add_argument(
        "--method", choices=["shap", "lime", "pdp", "ale"],
        default="shap",
    )
    method = parser.parse_args().method
    x_train, x_test, y_train, _, class_names = prepare_data()

    # Step2、训练与四种解释方法共用的 LightGBM 分类模型。
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(x_train, y_train)

    # Step3、每次选一种解释方式，核对其解释的是总体还是单个样本。
    if method == "shap":
        show_shap(model, x_train, x_test)
    elif method == "lime":
        show_lime(model, x_train, x_test, class_names)
    elif method == "pdp":
        show_pdp(model, x_train)
    else:
        show_ale(model, x_train)


if __name__ == "__main__":
    main()
