# -*- coding: utf-8 -*-
# 文件用途：比较列表、集合、字典、NumPy 数组和稀疏矩阵在典型操作中的作用。
# 对应章节/小节：第 8 章 8.6.2.1 数据结构优化示例
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.6、代码优化/8.6.2.1_data_structures.py"。
# 备注：计时结果受数据规模和机器状态影响；集合去重不保证原始顺序。

from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix


def ordered_unique_with_list(values):
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def main():
    values = list(range(1000)) * 5

    # Step1、比较列表逐个查找与集合去重；两种结果排序后应相同。
    started = perf_counter()
    list_result = ordered_unique_with_list(values)
    list_seconds = perf_counter() - started
    started = perf_counter()
    set_result = set(values)
    set_seconds = perf_counter() - started
    assert sorted(list_result) == sorted(set_result)
    print(f"列表去重：{list_seconds:.6f} 秒")
    print(f"集合去重：{set_seconds:.6f} 秒")

    # Step2、重复按用户 ID 查找时，比较列表扫描与字典索引。
    users = [{"id": index, "score": index / 10} for index in range(5000)]
    lookup_ids = [4999, 3000, 1000] * 100
    started = perf_counter()
    by_list = [
        next(user["score"] for user in users if user["id"] == key)
        for key in lookup_ids
    ]
    list_lookup_seconds = perf_counter() - started
    started = perf_counter()
    index = {user["id"]: user["score"] for user in users}
    by_dict = [index[key] for key in lookup_ids]
    dict_lookup_seconds = perf_counter() - started
    assert by_list == by_dict
    print(f"列表重复查找：{list_lookup_seconds:.6f} 秒")
    print(f"建字典并查找：{dict_lookup_seconds:.6f} 秒")

    # Step3、NumPy 适合矩阵运算，CSR 适合存储大量零值。
    rng = np.random.default_rng(42)
    left = rng.random((50, 50))
    right = rng.random((50, 50))
    product = left @ right
    print("矩阵乘积形状：", product.shape)
    dense = rng.choice([0.0, 1.0], size=(1000, 1000), p=[0.99, 0.01])
    sparse = csr_matrix(dense)
    sparse_bytes = sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
    print(f"密集矩阵：{dense.nbytes} 字节；CSR：{sparse_bytes} 字节")


if __name__ == "__main__":
    main()
