# -*- coding: utf-8 -*-
# 文件用途：对比串行与多进程处理独立计算任务。
# 对应章节/小节：第 8 章 8.6.2.4 并行计算优化示例
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.6、代码优化/8.6.2.4_T1_multiprocessing.py"。
# 备注：进程启动与数据传输有额外开销，小任务可能比串行更慢。

from multiprocessing import Pool
from os import cpu_count
from time import perf_counter


def sum_of_squares(chunk):
    """计算一个数据分块的平方和。"""
    return sum(value * value for value in chunk)


def main():
    # Step1、把相同输入分成独立任务，确保两种方案结果可比。
    values = list(range(200000))
    workers = min(4, cpu_count() or 1)
    chunk_size = (len(values) + workers - 1) // workers
    chunks = [
        values[index : index + chunk_size]
        for index in range(0, len(values), chunk_size)
    ]

    # Step2、串行执行并记录时间。
    started = perf_counter()
    serial_result = sum(sum_of_squares(chunk) for chunk in chunks)
    serial_seconds = perf_counter() - started

    # Step3、由进程池处理相同的分块，并验证结果一致。
    # T1、单机多进程方案：让进程池并行处理独立任务。
    started = perf_counter()
    with Pool(processes=workers) as pool:
        parallel_result = sum(pool.map(sum_of_squares, chunks))
    parallel_seconds = perf_counter() - started
    assert serial_result == parallel_result
    print(f"串行：{serial_seconds:.6f} 秒")
    print(f"{workers} 进程：{parallel_seconds:.6f} 秒")


if __name__ == "__main__":
    main()
