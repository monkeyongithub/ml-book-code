# -*- coding: utf-8 -*-
# 文件用途：比较线性、二次和四次多项式回归的拟合与泛化差异。
# 对应章节/小节：第 8 章 8.2.3 代码实战
# 作者：NYY
# 创建日期：2024-08-14
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.2、模型过拟合欠拟合问题/8.2.3_T1_model_complexity.py"。
# 备注：R² 同时在训练集和测试集上计算；小样本结果只用于观察模型复杂度。

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def main():
    # T1、通过多项式阶数比较模型复杂度。
    # Step1、准备书稿中的训练点和独立测试点。
    x_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
    y_train = np.array([7, 9, 13, 17.5, 18])
    x_test = np.array([6, 8, 11, 16]).reshape(-1, 1)
    y_test = np.array([8, 12, 15, 18])
    x_curve = np.linspace(5, 19, 100).reshape(-1, 1)

    # Step2、只改变多项式阶数，使三个模型的比较条件一致。
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, degree, title in zip(
        axes, (1, 2, 4), ("线性模型", "二次模型", "四次模型")
    ):
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )
        model.fit(x_train, y_train)
        train_r2 = r2_score(y_train, model.predict(x_train))
        test_r2 = r2_score(y_test, model.predict(x_test))
        print(f"{title}：训练 R²={train_r2:.3f}，测试 R²={test_r2:.3f}")

        # Step3、结合训练点、测试点与拟合曲线理解欠拟合和过拟合。
        ax.scatter(x_train, y_train, label="训练点")
        ax.scatter(x_test, y_test, marker="x", label="测试点")
        ax.plot(x_curve, model.predict(x_curve), label="预测曲线")
        ax.set_title(f"{title}：训练 R²={train_r2:.2f}，测试 R²={test_r2:.2f}")
        ax.set_xlabel("输入特征")
        ax.legend()
    axes[0].set_ylabel("目标值")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
