# -*- coding: utf-8 -*-
# 文件用途：演示预处理器在训练集拟合，避免测试数据泄露。
# 对应章节/小节：第 8 章 8.3.4 数据泄露及其优化
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.3、数据层面优化/8.3.4_prevent_data_leakage.py"。
# 备注：极端测试值用于放大两种拟合方式的差异，属于概念示例。

import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    # Step1、先明确训练集和测试集的边界。
    train = np.array([[1.0], [2.0], [3.0]])
    test = np.array([[1000.0]])

    # Step2、正确做法：仅用训练集估计均值和标准差。
    train_only_scaler = StandardScaler().fit(train)
    transformed_test = train_only_scaler.transform(test)
    print(f"训练集均值：{train_only_scaler.mean_[0]:.2f}")
    print(f"正确标准化后的测试值：{transformed_test[0, 0]:.2f}")

    # Step3、反例：全量数据参与拟合，会把测试集信息带入预处理器。
    leaky_scaler = StandardScaler().fit(np.vstack([train, test]))
    print(f"泄露时使用的均值：{leaky_scaler.mean_[0]:.2f}")
    print(f"泄露后标准化的测试值：{leaky_scaler.transform(test)[0, 0]:.2f}")


if __name__ == "__main__":
    main()
