# -*- coding: utf-8 -*-
# 文件用途：以 Boston 房价回归示例比较 MSE、RMSE、MAE、MAPE 和 R²。
# 对应章节/小节：第 6 章 6.3.1 模型评估：回归任务
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.3、模型评估与调优/6.3.1_regression_metrics.py
# 备注：MAPE 对接近零的真实值敏感；须结合其他误差指标理解。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shared.dataset import boston_df


def main():
    # Step1、按特征类型准备 Boston 数据；先切分再拟合预处理器。
    frame, target_name = boston_df()
    labels = pd.to_numeric(frame[target_name], errors="raise")
    features = frame.drop(columns=target_name)
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [name for name in features if name not in numeric]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42,
    )
    transformers = [
        ("numeric", SimpleImputer(strategy="median"), numeric),
    ]
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

    # Step2、在训练集拟合，在测试集获得连续预测值。
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    # Step3、分别观察平方误差、绝对误差、百分比误差与解释度。
    mse = mean_squared_error(y_test, predicted)
    print(f"MSE：{mse:.3f}")
    print(f"RMSE：{np.sqrt(mse):.3f}")
    print(f"MAE：{mean_absolute_error(y_test, predicted):.3f}")
    print(f"MAPE：{mean_absolute_percentage_error(y_test, predicted):.3f}")
    print(f"R²：{r2_score(y_test, predicted):.3f}")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_test, predicted, alpha=0.65)
    minimum = min(y_test.min(), predicted.min())
    maximum = max(y_test.max(), predicted.max())
    axes[0].plot([minimum, maximum], [minimum, maximum], "r--")
    axes[0].set(xlabel="真实房价", ylabel="预测房价", title="预测值与真实值")
    axes[1].hist(y_test.to_numpy() - predicted, bins=20)
    axes[1].set(xlabel="残差", ylabel="样本数", title="误差分布")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
