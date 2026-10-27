# -*- coding: utf-8 -*-
# 文件用途：用两组模拟业务指标演示 A/B 测试的显著性检验。
# 对应章节/小节：第 7 章 7.2.2 模型发布代码实战
# 作者：NYY
# 创建日期：2024-03-22
# 运行方式：从仓库根目录执行 python 七、模型发布、部署与监控/7.2、模型发布/7.2.2_ab_test.py
# 备注：模拟独立用户组的效果指标，不代表真实流量分配或模型上线效果。

import numpy as np
from scipy import stats


def compare_ab_groups(group_a, group_b, alpha=0.05):
    """比较两个独立用户组的平均指标，并给出均值差的置信区间。"""
    values_a = np.asarray(group_a, dtype=float)
    values_b = np.asarray(group_b, dtype=float)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError("两组指标都必须是一维数组。")
    if len(values_a) < 2 or len(values_b) < 2:
        raise ValueError("每组至少需要两个观测值。")
    if not np.isfinite(values_a).all() or not np.isfinite(values_b).all():
        raise ValueError("指标中不能包含 NaN 或无穷值。")
    if not 0 < alpha < 1:
        raise ValueError("显著性水平 alpha 必须介于 0 和 1 之间。")

    # Step2、对独立两组使用 Welch t 检验，置信区间采用相同的方差假设。
    n_a, n_b = len(values_a), len(values_b)
    var_a = np.var(values_a, ddof=1)
    var_b = np.var(values_b, ddof=1)
    variance_a = var_a / n_a
    variance_b = var_b / n_b
    standard_error = np.sqrt(variance_a + variance_b)
    if standard_error == 0:
        raise ValueError("两组指标均无波动，无法计算 t 检验。")

    t_statistic, p_value = stats.ttest_ind(
        values_b, values_a, equal_var=False
    )
    degrees_of_freedom = (variance_a + variance_b) ** 2 / (
        variance_a**2 / (n_a - 1) + variance_b**2 / (n_b - 1)
    )
    mean_difference = np.mean(values_b) - np.mean(values_a)
    margin = stats.t.ppf(1 - alpha / 2, degrees_of_freedom) * standard_error

    return {
        "mean_difference": mean_difference,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "confidence_interval": (mean_difference - margin, mean_difference + margin),
        "significant": p_value < alpha,
    }


def main():
    # Step1、模拟互不重叠的 A、B 两组用户效果指标。
    generator = np.random.default_rng(0)
    group_a = generator.normal(loc=10.0, scale=2.0, size=1000)
    group_b = generator.normal(loc=10.5, scale=2.0, size=1000)

    result = compare_ab_groups(group_a, group_b)

    # Step3、结合 p 值与均值差置信区间理解两组结果。
    print(f"均值差（B - A）：{result['mean_difference']:.4f}")
    print(f"t 统计量：{result['t_statistic']:.4f}")
    print(f"p 值：{result['p_value']:.4f}")
    lower, upper = result["confidence_interval"]
    print(f"95% 置信区间：({lower:.4f}, {upper:.4f})")
    print(f"是否达到显著性水平：{result['significant']}")


if __name__ == "__main__":
    main()
