# -*- coding: utf-8 -*-
# 文件用途：比较类型缩减、压缩存储与稀疏矩阵的空间占用。
# 对应章节/小节：第 8 章 8.3.5 数据降内存
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.3、数据层面优化/8.3.5_T1_storage_and_sparsity.py"。
# 备注：Parquet 文件是本脚本的输出；压缩率随数据分布改变。

import gzip
from pathlib import Path

import blosc
import numpy as np
import pandas as pd
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = (
    REPO_ROOT
    / "八、模型项目整体性分析、反思与优化"
    / "8.3、数据层面优化"
    / "data.parquet"
)


def main():
    # T1、从数值类型、压缩格式和稀疏结构观察存储空间。
    # Step1、生成可重复的低基数矩阵，并比较 float64 与 float32 的字节数。
    rng = np.random.default_rng(42)
    data = rng.integers(0, 10, size=(500, 500)).astype(np.float64)
    compact = data.astype(np.float32)
    print(f"float64 字节数：{data.nbytes}")
    print(f"float32 字节数：{compact.nbytes}")

    # Step2、比较同一份 float32 数据的两种无损字节压缩结果。
    raw = compact.tobytes()
    print(f"gzip 压缩字节数：{len(gzip.compress(raw))}")
    print(
        "blosc 压缩字节数："
        f"{len(blosc.compress(raw, typesize=compact.itemsize))}"
    )
    pd.DataFrame(compact).to_parquet(PARQUET_PATH, compression="gzip")
    print(f"Parquet 文件字节数：{PARQUET_PATH.stat().st_size}")

    # Step3、把大部分元素置零，比较密集与 CSR 稀疏存储。
    sparse_data = compact.copy()
    sparse_data[rng.random(sparse_data.shape) > 0.1] = 0
    csr = sparse.csr_matrix(sparse_data)
    csr_bytes = csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    print(f"密集矩阵字节数：{sparse_data.nbytes}")
    print(f"CSR 矩阵字节数：{csr_bytes}")


if __name__ == "__main__":
    main()
