# -*- coding: utf-8 -*-
# 文件用途：对比逐元素循环与向量化，以及冒泡排序与快速排序。
# 对应章节/小节：第 8 章 8.6.2.2 算法优化示例
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.6、代码优化/8.6.2.2_algorithm_choices.py"。
# 备注：两组方法分别使用同一份输入并核对结果；计时不保证在所有机器上得到同样的倍数。

from time import perf_counter

import numpy as np


def bubble_sort(values):
    result = values.copy()
    for end in range(len(result) - 1, 0, -1):
        for index in range(end):
            if result[index] > result[index + 1]:
                result[index], result[index + 1] = result[index + 1], result[index]
    return result


def main():
    # Step1、生成两组矩阵，后续每组比较都使用相同输入。
    rng = np.random.default_rng(42)
    left = rng.random((300, 300))
    right = rng.random((300, 300))

    # Step2、对同一组矩阵比较两种加法写法。
    # T1、逐元素循环计算矩阵加法。
    started = perf_counter()
    loop_sum = np.empty_like(left)
    for row in range(left.shape[0]):
        for column in range(left.shape[1]):
            loop_sum[row, column] = left[row, column] + right[row, column]
    loop_seconds = perf_counter() - started
    # T2、NumPy 向量化计算同一矩阵加法。
    started = perf_counter()
    vector_sum = left + right
    vector_seconds = perf_counter() - started
    np.testing.assert_allclose(loop_sum, vector_sum)
    print(f"逐元素循环：{loop_seconds:.6f} 秒")
    print(f"NumPy 向量化：{vector_seconds:.6f} 秒")

    # Step3、另取同一数组，比较两种排序写法并核对结果。
    values = rng.random(300)
    # T1、用 Python 循环执行冒泡排序。
    started = perf_counter()
    slow_sorted = bubble_sort(values.tolist())
    bubble_seconds = perf_counter() - started
    # T2、用 NumPy 执行快速排序。
    started = perf_counter()
    fast_sorted = np.sort(values, kind="quicksort")
    quick_seconds = perf_counter() - started
    np.testing.assert_allclose(slow_sorted, fast_sorted)
    print(f"冒泡排序：{bubble_seconds:.6f} 秒")
    print(f"快速排序：{quick_seconds:.6f} 秒")


if __name__ == "__main__":
    main()
