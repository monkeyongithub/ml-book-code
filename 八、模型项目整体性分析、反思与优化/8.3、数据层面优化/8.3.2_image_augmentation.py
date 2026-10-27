# -*- coding: utf-8 -*-
# 文件用途：以 MNIST 图像演示旋转、平移等训练数据增强操作。
# 对应章节/小节：第 8 章 8.3.2 数据增强
# 作者：NYY
# 创建日期：2024-03-22
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.3、数据层面优化/8.3.2_image_augmentation.py"。
# 备注：增强只应用于训练样本；MNIST 首次加载可能需要下载。

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    # Step1、载入训练图像，保留测试集不做增强。
    (train_images, train_labels), _ = mnist.load_data()
    originals = train_images[:9].astype("float32").reshape(9, 28, 28, 1) / 255
    labels = train_labels[:9]

    # Step2、只对训练样本执行随机旋转、平移、错切和缩放。
    augmenter = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
    )
    augmented, _ = next(
        augmenter.flow(originals, labels, batch_size=9, shuffle=False, seed=42)
    )

    # Step3、并排观察同一标签在增强前后的图像变化。
    fig, axes = plt.subplots(2, 9, figsize=(12, 3))
    for index in range(9):
        axes[0, index].imshow(originals[index, :, :, 0], cmap="gray")
        axes[1, index].imshow(augmented[index, :, :, 0], cmap="gray")
        axes[0, index].set_title(str(labels[index]))
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("原图")
    axes[1, 0].set_ylabel("增强后")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
