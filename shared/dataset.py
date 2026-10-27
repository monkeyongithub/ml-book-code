# -*- coding: utf-8 -*-
# 文件用途：为多个章节提供统一的演示数据集加载函数。
# 对应章节/小节：第 2、3、5、6、8 章的相关数据示例
# 作者：NYY
# 创建日期：2024-08-17
# 运行方式：由各章脚本通过 from shared.dataset import ... 导入，不单独运行。
# 备注：OpenML 和 UCI 数据集需在使用相应函数时联网；信用卡数据从第 3 章本地文件读取。

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_openml_frame(name: str) -> tuple[pd.DataFrame, str]:
    """从 OpenML 加载数据，并将目标列加入 DataFrame。"""
    from sklearn.datasets import fetch_openml

    # Step1、获取原始数据，复制特征列以避免修改下载对象。
    dataset = fetch_openml(name, version=1, as_frame=True)
    frame = dataset.data.copy()

    # Step2、附加目标列，使各章示例使用相同的数据结构。
    label_name = "target"
    frame[label_name] = dataset.target
    frame.info()
    return frame, label_name


def titanic_df() -> tuple[pd.DataFrame, str]:
    """加载 Titanic 生存预测数据。"""
    return _load_openml_frame("titanic")


def boston_df() -> tuple[pd.DataFrame, str]:
    """加载 Boston 房价预测数据。"""
    return _load_openml_frame("boston")


def iris_df() -> tuple[pd.DataFrame, str]:
    """加载 Iris 鸢尾花分类数据。"""
    return _load_openml_frame("iris")


def uci_car_df() -> tuple[pd.DataFrame, str]:
    """加载 UCI 汽车评价数据。"""
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "car/car.data"
    )
    columns = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class",
    ]

    # Step1、按数据集的公开字段顺序读取没有表头的 CSV。
    frame = pd.read_csv(url, names=columns)
    frame.info()
    return frame, "class"


def person_info_df() -> pd.DataFrame:
    """构造含异常年龄和身高的微型数据集。"""
    # Step1、保留异常值，供数据清洗和特征工程章节演示。
    return pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 185],
            "Height": [
                160, 165, 170, 175, 180, 185, 190,
                195, 200, 205, 210, 215, 5,
            ],
        }
    )


def creditcard_df() -> tuple[pd.DataFrame, str]:
    """读取第 3 章随书提供的信用卡欺诈数据。"""
    path = REPO_ROOT / "三、数据科学任务完整流程" / "creditcard.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少第 3 章数据文件：{path}")

    # Step1、从仓库内固定位置读取，不依赖启动脚本时的当前目录。
    frame = pd.read_csv(path)
    frame.info()
    return frame, "Class"
