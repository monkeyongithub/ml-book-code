# -*- coding: utf-8 -*-
# 文件用途：用八象空间标注八类模拟客户画像。
# 对应章节/小节：第 2 章 2.3.2 三维标签图—八象空间三维图
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 二、数据可视化技术/2.3、三维图/2.3.2_octant_labels.py
# 备注：坐标和客户类别为示意数据，用于理解三维标签的位置。

import matplotlib.pyplot as plt

CUSTOMER_LABELS = (
    "重要价值客户", "重要发展客户", "重要挽留客户", "重要保持客户",
    "一般价值客户", "一般发展客户", "一般挽留客户", "一般保持客户",
)


def main():
    # Step1、为三维空间八个正负组合分配书稿中的客户画像标签。
    positions = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    coordinates = [(x, y, z) for z in (1, -1) for x, y in positions]
    points = dict(zip(CUSTOMER_LABELS, coordinates))

    # Step2、画出相交于原点的三个坐标轴，显示八象空间边界。
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(projection="3d")
    for x, y, z in (
        ([-1.4, 1.4], [0, 0], [0, 0]),
        ([0, 0], [-1.4, 1.4], [0, 0]),
        ([0, 0], [0, 0], [-1.4, 1.4]),
    ):
        axis.plot(x, y, z, linestyle="--", color="gray", linewidth=1)

    # Step3、逐点添加散点与标签，对照书稿理解客户类别的空间位置。
    for label, (x, y, z) in points.items():
        color = "tab:red" if z > 0 else "tab:blue"
        axis.scatter(x, y, z, color=color, s=70)
        axis.text(x, y, z, label, color=color, fontsize=9)
    axis.set(xlabel="X", ylabel="Y", zlabel="Z")
    axis.set_xlim(-1.5, 1.5)
    axis.set_ylim(-1.5, 1.5)
    axis.set_zlim(-1.5, 1.5)
    axis.set_title("八象空间客户画像示意")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
