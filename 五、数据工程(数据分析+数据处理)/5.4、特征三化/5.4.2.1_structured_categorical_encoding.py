# -*- coding: utf-8 -*-
# 文件用途：演示结构化类别特征的独热、序数、目标与哈希编码。
# 对应章节/小节：第 5 章 5.4.2.1 狭义的类别型特征编码
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.4、特征三化/5.4.2.1_structured_categorical_encoding.py"
# 备注：目标编码只作说明；用于建模时须在训练集内拟合。

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from shared.dataset import titanic_df as load_titanic


def hash_category(value, buckets=6):
    # 将类别映射到固定数量的哈希桶；不同类别可能发生碰撞。
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def main():
    # Step1、读取书稿采用的 Titanic 类别特征和二分类目标。
    data, _ = load_titanic()
    data = data[["pclass", "sex", "embarked", "target"]].copy()
    embarked = data["embarked"].astype("string").fillna("unknown")
    sex = data["sex"].astype("string").fillna("unknown")
    pclass = pd.to_numeric(data["pclass"]).fillna(data["pclass"].mode().iloc[0])
    labels = pd.to_numeric(data["target"]).astype(int)

    # Step2、对无需目标值的类别列，比较独热、序数和标签编码。
    # T1、独热编码：每个港口类别单独占一列。
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_ports = one_hot.fit_transform(embarked.to_frame())
    print("独热编码列：", one_hot.get_feature_names_out(["embarked"]))
    print("前五行：", encoded_ports[:5])

    # T2、序数编码：船舱等级有明确的 1、2、3 顺序。
    ordinal = OrdinalEncoder(categories=[[1, 2, 3]])
    print("序数编码：", ordinal.fit_transform(pclass.to_frame())[:5].ravel())

    # T3、标签编码：类别标签被映射为整数，不意味着类别之间有距离。
    label_encoder = LabelEncoder()
    encoded_sex = label_encoder.fit_transform(sex)
    print("标签编码映射：", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
    print("前五行：", encoded_sex[:5])

    # Step3、演示依赖目标值的编码与固定桶数的哈希编码。
    # T4、目标编码：只用训练集标签计算类别均值，再映射到留出的样本。
    train_index, test_index = train_test_split(
        data.index, test_size=0.2, stratify=labels, random_state=42
    )
    train_labels = labels.loc[train_index]
    target_means = train_labels.groupby(sex.loc[train_index]).mean()
    encoded_test = sex.loc[test_index].map(target_means).fillna(train_labels.mean())
    print("训练集目标均值：", target_means.to_dict())
    print("留出样本的目标编码：", encoded_test.head().to_list())

    # T5、哈希编码：固定桶数便于处理大类别集，同时观察可能的碰撞。
    hash_mapping = {category: hash_category(category) for category in embarked.unique()}
    print("哈希编码映射：", hash_mapping)


if __name__ == "__main__":
    main()
