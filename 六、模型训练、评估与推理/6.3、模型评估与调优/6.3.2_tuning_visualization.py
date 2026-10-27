# -*- coding: utf-8 -*-
# 文件用途：将 LightGBM 的学习率与树深度网格搜索结果绘成热力图和三维曲面。
# 对应章节/小节：第 6 章 6.3.2.2 模型调参方法
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.3、模型评估与调优/6.3.2_tuning_visualization.py
# 备注：使用训练集内部交叉验证调参，独立测试集仅用于最终评价。

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split


def main():
    # Step1、生成二分类数据，并留出不参与调参的测试集。
    features, labels = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=0, random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )

    # Step2、在训练集内对学习率和树深度进行三折网格搜索。
    learning_rates = [0.03, 0.1, 0.2]
    depths = [2, 4, 6]
    search = GridSearchCV(
        LGBMClassifier(n_estimators=60, verbose=-1, random_state=42),
        {"learning_rate": learning_rates, "max_depth": depths},
        scoring="roc_auc", cv=3,
    )
    search.fit(x_train, y_train)
    print(f"最佳参数：{search.best_params_}")
    print(f"交叉验证 ROC AUC：{search.best_score_:.3f}")

    # Step3、将每一组参数的交叉验证结果放回二维网格，再绘制结果。
    scores = np.full((len(depths), len(learning_rates)), np.nan)
    for params, score in zip(
        search.cv_results_["params"], search.cv_results_["mean_test_score"]
    ):
        row = depths.index(params["max_depth"])
        column = learning_rates.index(params["learning_rate"])
        scores[row, column] = score
    if np.isnan(scores).any():
        raise RuntimeError("网格搜索结果不完整，无法绘图。")

    figure = plt.figure(figsize=(12, 4))
    heatmap = figure.add_subplot(121)
    image = heatmap.imshow(scores, origin="lower", cmap="viridis")
    heatmap.set_xticks(range(len(learning_rates)), learning_rates)
    heatmap.set_yticks(range(len(depths)), depths)
    heatmap.set(
        xlabel="学习率", ylabel="树深度", title="交叉验证 ROC AUC 热力图"
    )
    figure.colorbar(image, ax=heatmap)

    surface = figure.add_subplot(122, projection="3d")
    x_grid, y_grid = np.meshgrid(learning_rates, depths)
    surface.plot_surface(x_grid, y_grid, scores, cmap="viridis")
    surface.set(
        xlabel="学习率", ylabel="树深度", zlabel="ROC AUC",
        title="网格搜索结果曲面",
    )
    figure.tight_layout()
    plt.show()

    # Step4、只用选中的模型在独立测试集上评价一次。
    probability = search.best_estimator_.predict_proba(x_test)[:, 1]
    print(f"测试集 ROC AUC：{roc_auc_score(y_test, probability):.3f}")


if __name__ == "__main__":
    main()
