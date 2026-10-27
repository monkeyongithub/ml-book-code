# -*- coding: utf-8 -*-
# 文件用途：演示释放引用、生成器和稀疏独热编码的内存使用特点。
# 对应章节/小节：第 8 章 8.6.2.5 内存空间优化示例
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.6、代码优化/8.6.2.5_memory_usage.py"。
# 备注：getsizeof 只统计对象本身的浅层大小；删除引用不等于保证操作系统立即回收内存。

import sys

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def main():
    # Step1、不再需要大数组时删除引用，避免后续步骤继续持有它。
    large_array = np.ones(1_000_000, dtype=np.float64)
    print(f"数组数据区：{large_array.nbytes} 字节")
    average = large_array.mean()
    del large_array
    print(f"保留下来的均值：{average:.1f}")

    # Step2、比较列表与生成器的对象浅层大小及计算结果。
    # T1、列表一次保存所有值。
    values = list(range(100000))
    # T2、生成器按需给出值。
    generator = (value for value in range(100000))
    print(f"列表对象浅层大小：{sys.getsizeof(values)} 字节")
    print(f"生成器对象浅层大小：{sys.getsizeof(generator)} 字节")
    assert sum(values) == sum(generator)

    # Step3、用同一分类输入比较密集与稀疏独热编码的数据区大小。
    # T1、生成密集独热编码。
    categories = np.array([str(index) for index in range(1000)]).reshape(-1, 1)
    dense = OneHotEncoder(sparse_output=False).fit_transform(categories)
    # T2、生成稀疏独热编码。
    sparse = OneHotEncoder(sparse_output=True).fit_transform(categories)
    sparse_bytes = sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
    print(f"密集编码数据区：{dense.nbytes} 字节")
    print(f"稀疏编码数据区：{sparse_bytes} 字节")


if __name__ == "__main__":
    main()
