# -*- coding: utf-8 -*-
# 文件用途：以两类样本数量的变化说明过采样的作用。
# 对应章节/小节：第 3 章 3.2.5.2 过采样的概述及其常用方法
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 三、数据科学任务完整流程/3.2、数据认知与数据收集/3.2.5、数据不均衡/3.2.5.2_oversampling_illustration.py
# 备注：仅演示类别数量变化；具体算法见 3.2.5_sampling_methods.py。

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Step1、构造多数类与少数类数量差异明显的示意数据。
    categories = ["多数类", "少数类"]
    before = [8, 2]

    # Step2、只增加少数类数量，展示过采样后的均衡效果。
    after = [8, 8]
    positions = np.arange(len(categories))
    width = 0.35

    # Step3、并排比较处理前后的各类样本数。
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(positions - width / 2, before, width, label="过采样前")
    axis.bar(positions + width / 2, after, width, label="过采样后")
    axis.set_xticks(positions, categories)
    axis.set_ylabel("样本数")
    axis.set_title("过采样前后的类别数量")
    axis.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
