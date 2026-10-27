# -*- coding: utf-8 -*-
# 文件用途：演示按数值范围和重复率缩减 DataFrame 的内存占用。
# 对应章节/小节：第 8 章 8.3.5 数据降内存
# 作者：NYY
# 创建日期：2024-11-29
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.3、数据层面优化/8.3.5_T2_dataframe_downcasting.py"。
# 备注：浮点降精度仅在指定误差范围内进行；函数返回副本，不修改输入数据。

import numpy as np
import pandas as pd


def reduce_memory(frame: pd.DataFrame) -> pd.DataFrame:
    """按列选择更紧凑且适合当前数据的类型。"""
    result = frame.copy()
    for column in result.columns:
        series = result[column]
        if pd.api.types.is_integer_dtype(series):
            # Step1、整数列按实际取值范围降位宽。
            result[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            # Step2、浮点列仅在近似值可接受时转为 float32。
            candidate = series.astype(np.float32)
            if np.allclose(series, candidate, rtol=1e-6, atol=1e-9, equal_nan=True):
                result[column] = candidate
        elif series.dtype == object and series.nunique(dropna=False) < len(series) / 2:
            # Step3、低基数字符串列改用 category。
            result[column] = series.astype("category")
    return result


def main():
    frame = pd.DataFrame(
        {
            "count": np.arange(1000, dtype=np.int64),
            "score": np.linspace(0, 1, 1000, dtype=np.float64),
            "group": ["A", "B", "C", "D"] * 250,
        }
    )
    # T2、按各列实际取值缩减 DataFrame 的内存占用。
    reduced = reduce_memory(frame)
    before = frame.memory_usage(deep=True).sum()
    after = reduced.memory_usage(deep=True).sum()
    print(f"优化前：{before} 字节")
    print(f"优化后：{after} 字节")
    print(reduced.dtypes)


if __name__ == "__main__":
    main()
