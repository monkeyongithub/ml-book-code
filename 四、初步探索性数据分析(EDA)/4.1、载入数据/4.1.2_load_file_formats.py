# -*- coding: utf-8 -*-
# 文件用途：对照书稿演示读取 TXT、CSV、XLSX、JSON、XML、Parquet 和 HDF5 文件。
# 对应章节/小节：第 4 章 4.1.2 载入数据代码实战
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "四、初步探索性数据分析(EDA)/4.1、载入数据/4.1.2_load_file_formats.py" <文件路径>
# 备注：将资源路径作为参数传入；HDF5 默认读取名为 data 的原生数据集或 pandas 键。

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import pandas as pd
import pyarrow.parquet as parquet


def read_txt(path):
    """按 UTF-8 文本读取 TXT 文件。"""
    return path.read_text(encoding="utf-8")


def read_csv(path):
    """读取 CSV 表格。"""
    return pd.read_csv(path)


def read_xlsx(path):
    """读取 XLSX 表格。"""
    return pd.read_excel(path)


def read_json(path):
    """读取 JSON 对象或数组。"""
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_xml(path):
    """读取 XML 并返回根节点。"""
    return ET.parse(path).getroot()


def read_parquet(path):
    """读取 Parquet 表格；也可使用 pandas.read_parquet。"""
    return parquet.read_table(path).to_pandas()


def read_hdf5(path, dataset_name="data"):
    """读取原生 HDF5 数据集，或 pandas 写入的 HDFStore 表。"""
    with h5py.File(path, "r") as source:
        if dataset_name not in source:
            raise KeyError(f"HDF5 文件中不存在数据集或键：{dataset_name}")
        node = source[dataset_name]
        if isinstance(node, h5py.Dataset):
            return node[:]  # 与书稿中的原生 HDF5 数据集示例对应。
        is_pandas_group = isinstance(node, h5py.Group) and "pandas_type" in node.attrs

    if is_pandas_group:
        # pandas 的 HDFStore 把表保存在 Group 中，需按键还原 DataFrame。
        return pd.read_hdf(path, key=dataset_name)
    raise TypeError(f"{dataset_name} 是 HDF5 组，但不是可读取的 pandas 表")


READERS = {
    ".txt": read_txt,
    ".csv": read_csv,
    ".xlsx": read_xlsx,
    ".json": read_json,
    ".xml": read_xml,
    ".parquet": read_parquet,
    ".h5": read_hdf5,
    ".hdf5": read_hdf5,
}


def main():
    parser = argparse.ArgumentParser(description="按文件格式读取数据。")
    parser.add_argument("file", type=Path, help="相对于仓库根目录的数据文件路径")
    parser.add_argument("--dataset", default="data", help="HDF5 内部数据集名称或 pandas 键")
    args = parser.parse_args()

    # Step1、根据扩展名选择与书稿示例对应的读取方式。
    path = args.file
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        parser.error(f"不支持的文件格式：{path.suffix}")
    if not path.is_file():
        parser.error(f"文件不存在：{path}")

    # Step2、读取数据；HDF5 可按原生数据集名称或 pandas 键定位。
    result = reader(path, args.dataset) if reader is read_hdf5 else reader(path)

    # Step3、展示读取结果的局部内容，避免直接打印整份数据。
    if isinstance(result, pd.DataFrame):
        print(result.head())
    elif isinstance(result, ET.Element):
        print(f"XML 根节点：{result.tag}；子节点数：{len(result)}")
    else:
        print(str(result)[:500])


if __name__ == "__main__":
    main()
