# -*- coding: utf-8 -*-
# 文件用途：演示将处理后的特征表导出为常用文件与 SQLite 格式。
# 对应章节/小节：第 5 章 5.6 特征导出
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.6、特征导出/5.6_feature_export.py"
# 备注：运行时在第 5 章 output 目录生成文件，不覆盖随书数据。

from pathlib import Path
import sqlite3

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from shared.dataset import titanic_df as load_titanic


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "五、数据工程(数据分析+数据处理)" / "output"


def prepare_features():
    # Step1、选择书稿使用的 Titanic 特征和目标变量。
    source, _ = load_titanic()
    data = source[
        ["pclass", "sex", "age", "sibsp", "fare", "embarked", "target"]
    ].copy()

    # Step2、数值列按中位数填充，类别列按众数填充，再做独热编码。
    numeric = ["pclass", "age", "sibsp", "fare"]
    categories = ["sex", "embarked"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data[numeric] = data[numeric].fillna(data[numeric].median())
    for column in categories:
        values = data[column].astype("string")
        data[column] = values.fillna(values.mode().iloc[0])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(data[categories])
    encoded_frame = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categories),
        index=data.index,
    )
    result = pd.concat([data.drop(columns=categories), encoded_frame], axis=1)
    result["target"] = pd.to_numeric(result["target"]).astype(int)
    return result


def export_data(data, file_format):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"titanic_df.{file_format}"
    if file_format == "csv":
        data.to_csv(path, index=False)
    elif file_format == "json":
        data.to_json(path, orient="records", force_ascii=False)
    elif file_format == "xlsx":
        data.to_excel(path, index=False)
    elif file_format == "parquet":
        data.to_parquet(path, index=False)
    elif file_format == "db":
        connection = sqlite3.connect(path)
        try:
            data.to_sql("titanic_df", connection, if_exists="replace", index=False)
        finally:
            connection.close()
    elif file_format == "h5":
        data.to_hdf(path, key="data", mode="w", format="table")
    else:
        raise ValueError(f"不支持的格式：{file_format}")
    print("已导出：", path.relative_to(REPO_ROOT))


def main():
    # Step3、分别导出书稿实际调用的六种格式，便于比较用途。
    data = prepare_features()
    for file_format in ("csv", "json", "xlsx", "parquet", "db", "h5"):
        export_data(data, file_format)


if __name__ == "__main__":
    main()
