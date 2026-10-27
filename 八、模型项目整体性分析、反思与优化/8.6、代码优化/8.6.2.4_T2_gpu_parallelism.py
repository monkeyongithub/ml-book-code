# -*- coding: utf-8 -*-
# 文件用途：比较同一张量在 CPU 和 CUDA GPU 上的计算耗时。
# 对应章节/小节：第 8 章 8.6.2.4 并行计算优化示例
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.6、代码优化/8.6.2.4_T2_gpu_parallelism.py"。
# 备注：无 CUDA 时跳过 GPU 对比；计时只覆盖加法计算，不包含数据传输。

from time import perf_counter

import torch


def main():
    if not torch.cuda.is_available():
        print("当前环境无可用 CUDA GPU，跳过 GPU 对比。")
        return

    # Step1、准备同一份 CPU 张量，并提前复制到 GPU。
    cpu_tensor = torch.rand((2048, 2048))
    gpu_tensor = cpu_tensor.to("cuda")
    torch.cuda.synchronize()

    # Step2、测量 CPU 上的张量加法，作为本脚本的基线。
    started = perf_counter()
    cpu_result = cpu_tensor + cpu_tensor
    cpu_seconds = perf_counter() - started

    # Step3、测量同一张量在 GPU 上的加法，并核对两端结果。
    # T2、CUDA GPU 方案：运算异步执行，同步完成后才停止计时。
    started = perf_counter()
    gpu_result = gpu_tensor + gpu_tensor
    torch.cuda.synchronize()
    gpu_seconds = perf_counter() - started

    # 核对两端的运算结果，再解释本机测得的时间。
    torch.testing.assert_close(cpu_result, gpu_result.cpu())
    print(f"CPU 计算：{cpu_seconds:.6f} 秒")
    print(f"GPU 计算：{gpu_seconds:.6f} 秒")


if __name__ == "__main__":
    main()
