# -*- coding: utf-8 -*-
# 文件用途：生成 Titanic 数据集的探索性分析报告，概览字段和数据质量。
# 对应章节/小节：第 4 章 4.2.2 初步概览数据集信息代码实战
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "四、初步探索性数据分析(EDA)/4.2、初步概览数据集信息/4.2.2_profile_dataset.py"
# 备注：首次获取 OpenML Titanic 数据可能需要网络；报告在本章目录下生成。

from pathlib import Path

from ydata_profiling import ProfileReport

from shared.dataset import titanic_df

REPORT_PATH = Path("四、初步探索性数据分析(EDA)/4.2、初步概览数据集信息/titanic_profile.html")


def main():
    # Step1、载入书稿采用的 Titanic 数据集。
    data, _ = titanic_df()

    # Step2、生成字段类型、缺失情况和分布的 EDA 报告。
    report = ProfileReport(data, title="Titanic 数据集概览")

    # Step3、保存报告，供浏览器查看图表和摘要。
    report.to_file(REPORT_PATH)
    print(f"报告已保存：{REPORT_PATH}")


if __name__ == "__main__":
    main()
