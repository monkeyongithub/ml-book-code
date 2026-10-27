# -*- coding: utf-8 -*-
# 文件用途：以 Boston 房价回归示例定位绝对误差较大的样本。
# 对应章节/小节：第 6 章 6.4.1 Bad-case 分析：回归任务
# 作者：NYY
# 创建日期：2024-08-14
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.4、模型预测结果剖析/6.4.1_regression_bad_cases.py
# 备注：大误差阈值是当前测试集绝对误差的 90% 分位数，仅用于观察。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shared.dataset import boston_df


def main():
    # Step1、准备 Boston 数据；预处理器只学习训练集。
    frame, target_name = boston_df()
    labels = pd.to_numeric(frame[target_name], errors="raise")
    features = frame.drop(columns=target_name)
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [name for name in features if name not in numeric]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42,
    )
    transformers = [("numeric", SimpleImputer(strategy="median"), numeric)]
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    model = Pipeline(
        [
            ("preprocessing", ColumnTransformer(transformers)),
            ("regressor", LGBMRegressor(
                n_estimators=100, random_state=42, verbose=-1
            )),
        ]
    )
    model.fit(x_train, y_train)

    # Step2、计算逐样本残差，而非只依赖整体平均指标。
    predicted = model.predict(x_test)
    residuals = y_test.to_numpy() - predicted
    absolute_errors = np.abs(residuals)
    threshold = np.quantile(absolute_errors, 0.90)
    large_error = absolute_errors >= threshold
    print(f"MAE：{mean_absolute_error(y_test, predicted):.3f}")
    print(f"RMSE：{np.sqrt(mean_squared_error(y_test, predicted)):.3f}")
    print(f"R²：{r2_score(y_test, predicted):.3f}")
    print(f"绝对误差较大的样本数：{np.count_nonzero(large_error)}")
    print(x_test.iloc[np.flatnonzero(large_error)].head())

    # Step3、对比真实值与预测值，并观察残差分布。
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_test, predicted, c=np.where(large_error, "red", "blue"))
    axes[0].set(xlabel="真实房价", ylabel="预测房价", title="大误差样本标红")
    axes[1].hist(residuals, bins=20)
    axes[1].set(xlabel="残差", ylabel="样本数", title="残差分布")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
