# -*- coding: utf-8 -*-
# 文件用途：用共同分箱和样本占比演示 PSI 数据分布偏移指标。
# 对应章节/小节：第 7 章 7.4.1 模型监控概述与稳定性指标
# 作者：NYY
# 创建日期：2024-03-22
# 运行方式：从仓库根目录执行 python 七、模型发布、部署与监控/7.4、模型监控/7.4.1_psi_distribution_shift.py
# 备注：书稿未给出可核验的 CSI 计算式，本示例只计算 PSI，不套用书稿的 CSI 阈值。

import numpy as np


def calculate_psi(reference, current, n_bins=10, smoothing=1e-6):
    """用基准样本确定分箱，计算两组样本的总体稳定性指数。"""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("两组样本都必须是一维数组。")
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("两组样本都不能为空。")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("样本中不能包含 NaN 或无穷值。")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("分箱数量至少为 2。")
    if smoothing <= 0:
        raise ValueError("平滑值必须大于 0。")

    # Step2、只用基准样本确定边界，并让两组样本共用同一组分箱。
    edges = np.histogram_bin_edges(reference, bins=n_bins)
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference, bins=edges)[0]
    current_counts = np.histogram(current, bins=edges)[0]

    # Step3、计算每个分箱的占比；平滑处理避免 log(0)。
    reference_share = (reference_counts + smoothing) / (
        len(reference) + smoothing * n_bins
    )
    current_share = (current_counts + smoothing) / (
        len(current) + smoothing * n_bins
    )
    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        )
    )


def main():
    # Step1、模拟基准分布、相近分布和明显偏移的分布。
    generator = np.random.default_rng(0)
    reference = generator.normal(loc=10.0, scale=2.0, size=1000)
    similar = generator.normal(loc=10.0, scale=2.0, size=1000)
    shifted = generator.normal(loc=12.0, scale=2.0, size=1000)

    print(f"相近分布的 PSI：{calculate_psi(reference, similar):.4f}")
    print(f"偏移分布的 PSI：{calculate_psi(reference, shifted):.4f}")


if __name__ == "__main__":
    main()
