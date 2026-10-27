# -*- coding: utf-8 -*-
# 文件用途：用同一份合成回归数据观察 L1 系数稀疏与 L2 系数收缩。
# 对应章节/小节：第 8 章 8.2.2 对比：L1 正则化、L2 正则化
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 python "八、模型项目整体性分析、反思与优化/8.2、模型过拟合欠拟合问题/8.2.2_l1_l2_regularization.py"。
# 备注：两种 alpha 仅用于展示各自作用，不代表相同惩罚强度；本例不比较泛化评分。

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


def main():
    # Step1、固定随机种子，构造只有三个有效特征的小型回归数据。
    generator = np.random.default_rng(42)
    features = generator.normal(size=(160, 8))
    true_coefficients = np.array([3.0, -2.0, 0.5, 0, 0, 0, 0, 0])
    target = features @ true_coefficients + generator.normal(
        scale=0.5, size=len(features)
    )

    # Step2、先标准化特征，再用无正则化模型提供系数参照。
    standardized = StandardScaler().fit_transform(features)
    baseline = LinearRegression().fit(standardized, target)

    # T1、L1：Lasso 可将一些不重要特征的系数压到零。
    l1_model = Lasso(alpha=0.45, max_iter=10000).fit(standardized, target)

    # T2、L2：Ridge 倾向于缩小系数幅度，而非主动产生稀疏解。
    l2_model = Ridge(alpha=80.0).fit(standardized, target)

    # Step3、对照系数和图形，只讨论稀疏与收缩，不据此判定模型优劣。
    for name, model in (
        ("无正则化", baseline),
        ("L1（Lasso）", l1_model),
        ("L2（Ridge）", l2_model),
    ):
        zero_count = np.count_nonzero(np.isclose(model.coef_, 0, atol=1e-10))
        print(f"{name}：系数={np.round(model.coef_, 3)}；零系数数目={zero_count}")

    positions = np.arange(features.shape[1])
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(positions - width, baseline.coef_, width, label="无正则化")
    axis.bar(positions, l1_model.coef_, width, label="L1（Lasso）")
    axis.bar(positions + width, l2_model.coef_, width, label="L2（Ridge）")
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set_xticks(positions, [f"特征{index + 1}" for index in positions])
    axis.set(ylabel="标准化后的特征系数", title="L1 稀疏与 L2 收缩")
    axis.legend()
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
