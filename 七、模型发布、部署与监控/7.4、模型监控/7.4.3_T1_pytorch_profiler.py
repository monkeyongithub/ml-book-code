# -*- coding: utf-8 -*-
# 文件用途：用 PyTorch Profiler 观察简单网络训练时的计算与内存开销。
# 对应章节/小节：第 7 章 7.4.3 模型监控代码实战
# 作者：NYY
# 创建日期：2024-03-20
# 运行方式：从仓库根目录执行 python 七、模型发布、部署与监控/7.4、模型监控/7.4.3_T1_pytorch_profiler.py
# 备注：随机数据仅用于性能分析，不用于评估模型预测效果。

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile


class SimpleNet(nn.Module):
    """书稿中的单层分类网络。"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, features):
        return self.fc(features)


def main():
    # T1、使用 PyTorch Profiler 分析训练步骤。
    # Step1、准备固定的示例数据，并按实际设备放置模型和张量。
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.randn(1000, 10, device=device)
    labels = torch.randint(0, 2, (1000,), device=device)
    model = SimpleNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Step2、仅在使用 CUDA 设备时记录 CUDA 活动。
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(10):
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

    # Step3、按 CPU 耗时展示各操作；CUDA 可用时也会记录其活动。
    print(f"分析设备：{device}")
    print(
        profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=5
        )
    )


if __name__ == "__main__":
    main()
