# -*- coding: utf-8 -*-
# 文件用途：演示损失值动态趋势图与示意点位的动态轨迹图。
# 对应章节/小节：第 2 章 2.4.1 动态趋势图、2.4.2 动态轨迹图
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 二、数据可视化技术/2.4、动态图/2.4_dynamic_visualizations.py --chart trend
# 备注：使用 --chart 选择 trend 或 trajectory；轨迹使用抽象点位，不涉及地理地图。

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animate_loss_trend():
    """逐轮绘制指数衰减的示意损失值。"""
    epochs = np.arange(1, 101)
    losses = 5 * np.exp(-0.035 * (epochs - 1))
    figure, axis = plt.subplots(figsize=(8, 5))
    (line,) = axis.plot([], [], color="tab:red", linewidth=2)
    axis.set(xlim=(1, epochs[-1]), ylim=(0, losses[0] + 0.5))
    axis.set(xlabel="训练轮次（Epoch）", ylabel="损失值（Loss）")
    axis.set_title("损失值随训练轮次的动态变化")

    def update(frame):
        line.set_data(epochs[:frame], losses[:frame])
        return (line,)

    return FuncAnimation(
        figure, update, frames=range(1, len(epochs) + 1),
        interval=60, blit=True, repeat=False,
    )


def animate_trajectory():
    """沿抽象点位逐步展示运动路径与当前位置。"""
    points = np.array([
        [0.12, 0.20], [0.32, 0.72], [0.58, 0.62],
        [0.78, 0.85], [0.88, 0.30],
    ])
    labels = ("A", "B", "C", "D", "E")
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(points[:, 0], points[:, 1], color="lightgray", s=70)
    for label, (x, y) in zip(labels, points):
        axis.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    (route,) = axis.plot([], [], color="tab:blue", linewidth=2)
    (current,) = axis.plot([], [], "o", color="tab:red", markersize=9)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="X", ylabel="Y")
    axis.set_title("示意点位的动态轨迹")
    axis.set_aspect("equal", adjustable="box")

    def update(frame):
        visited = points[:frame]
        route.set_data(visited[:, 0], visited[:, 1])
        current.set_data([visited[-1, 0]], [visited[-1, 1]])
        return route, current

    return FuncAnimation(
        figure, update, frames=range(1, len(points) + 1),
        interval=700, blit=True, repeat=False,
    )


def main():
    parser = argparse.ArgumentParser(description="选择书稿 2.4 节的一种动态图。")
    parser.add_argument(
        "--chart", choices=("trend", "trajectory"), default="trend"
    )
    chart = parser.parse_args().chart

    # Step1、按书稿小节选择训练趋势或平面轨迹。
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    # Step2、逐帧更新曲线或移动点，观察时间顺序如何进入静态坐标系。
    animation = (
        animate_loss_trend() if chart == "trend" else animate_trajectory()
    )

    # Step3、在动画窗口中观察变化过程，并保持动画对象存活至窗口关闭。
    plt.tight_layout()
    plt.show()
    return animation


if __name__ == "__main__":
    main()
