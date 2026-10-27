# -*- coding: utf-8 -*-
# 文件用途：按数据类型初步区分 Titanic 数据集中的数值、类别和时间特征。
# 对应章节/小节：第 4 章 4.3.2 划分特征类型代码实战
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "四、初步探索性数据分析(EDA)/4.3、划分特征类型/4.3.2_identify_feature_types.py"
# 备注：自动识别仅是初步判断，业务含义仍须结合书稿分析。

from shared.dataset import titanic_df


def identify_feature_types(data, target_name):
    """按 pandas 类型列出特征，不把目标变量算作输入特征。"""
    features = data.drop(columns=[target_name])
    return {
        "数值型": features.select_dtypes(include="number").columns.tolist(),
        "类别型": features.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist(),
        "日期时间型": features.select_dtypes(include="datetime").columns.tolist(),
    }


def main():
    # Step1、载入数据并识别目标变量。
    data, target_name = titanic_df()

    # Step2、按字段类型划分输入特征。
    feature_groups = identify_feature_types(data, target_name)

    # Step3、对照书稿中的四类特征，检查自动划分是否符合业务含义。
    for group_name, columns in feature_groups.items():
        print(f"{group_name}：{', '.join(columns) if columns else '无'}")
    print("提示：等级编号等数值字段可能在业务上属于类别特征。")


if __name__ == "__main__":
    main()
