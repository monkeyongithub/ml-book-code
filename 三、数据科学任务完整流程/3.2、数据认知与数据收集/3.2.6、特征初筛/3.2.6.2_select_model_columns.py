# -*- coding: utf-8 -*-
# 文件用途：在读取 Titanic CSV 时初步筛选用于建模的特征列。
# 对应章节/小节：第 3 章 3.2.6.2 特征初筛代码实战
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 三、数据科学任务完整流程/3.2、数据认知与数据收集/3.2.6、特征初筛/3.2.6.2_select_model_columns.py
# 备注：沿用仓库现有 Titanic CSV 的小写字段名；列的最终取舍仍需结合业务判断。

from pathlib import Path

import pandas as pd

CSV_PATH = Path("五、数据工程(数据分析+数据处理)/titanic_df.csv")
FEATURE_COLUMNS = ["pclass", "sex", "age", "sibsp", "fare"]
TARGET_COLUMN = "target"


def main():
    # Step1、按书稿中的初筛思路，列出本例要保留的特征和目标列。
    selected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"缺少示例数据：{CSV_PATH}")

    # Step2、用 usecols 在载入阶段筛选字段，避免读取无关列。
    data = pd.read_csv(CSV_PATH, usecols=selected_columns)

    # Step3、核对入模特征与目标列，理解初筛仅是建模前的第一步。
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    print(features.head())
    print(f"已选特征：{list(features.columns)}；目标列：{target.name}")


if __name__ == "__main__":
    main()
