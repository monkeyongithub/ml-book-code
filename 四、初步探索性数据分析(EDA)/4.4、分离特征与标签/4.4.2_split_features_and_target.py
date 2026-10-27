# -*- coding: utf-8 -*-
# 文件用途：演示用 drop、loc 和 CSV 的 usecols 分离输入特征与目标变量。
# 对应章节/小节：第 4 章 4.4.2 分离特征与标签代码实战
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "四、初步探索性数据分析(EDA)/4.4、分离特征与标签/4.4.2_split_features_and_target.py"
# 备注：T3 使用仓库内已有的 Titanic CSV 且只取指定列，输出维度不能与 T1/T2 直接比较。

from pathlib import Path

import pandas as pd

from shared.dataset import titanic_df

CSV_PATH = Path("五、数据工程(数据分析+数据处理)/titanic_df.csv")


def main():
    # Step1、载入书稿中的 Titanic 数据，并确定目标变量名称。
    data, target_name = titanic_df()

    # Step2、在同一份 DataFrame 上比较两种分离特征与标签的写法。
    # T1、使用 drop 排除目标列，保留所有输入特征。
    features_by_drop = data.drop(columns=[target_name])
    target_by_drop = data[target_name]
    print("T1：", features_by_drop.shape, target_by_drop.shape)

    # T2、使用 loc 的列条件选择输入特征。
    features_by_loc = data.loc[:, data.columns != target_name]
    target_by_loc = data.loc[:, target_name]
    print("T2：", features_by_loc.shape, target_by_loc.shape)

    # Step3、从仓库内 CSV 读取指定列，演示读取时筛选特征与标签。
    # T3、使用 usecols 读取部分列；输入与 T1/T2 不同，勿直接比较维度。
    selected_columns = ["pclass", "sex", "age", target_name]
    selected_data = pd.read_csv(CSV_PATH, usecols=selected_columns)
    features_by_usecols = selected_data.drop(columns=[target_name])
    target_by_usecols = selected_data[target_name]
    print("T3：", features_by_usecols.shape, target_by_usecols.shape)


if __name__ == "__main__":
    main()
