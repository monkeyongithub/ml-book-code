# -*- coding: utf-8 -*-
# 文件用途：比较简单随机抽样与按类别比例分层抽样的样本分布。
# 对应章节/小节：第 3 章 3.2.4.2 数据采样的常用方法
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 python 三、数据科学任务完整流程/3.2、数据认知与数据收集/3.2.4、数据采样/3.2.4.2_random_and_stratified_sampling.py
# 备注：总体与类别均为固定示意数据；分层只保证已知类别的样本比例。

from collections import Counter
from random import Random

SAMPLE_SIZE = 20
CATEGORIES = ("常见类", "少见类")


def show_distribution(name, records):
    """显示总体或样本中两类记录的数量与比例。"""
    counts = Counter(record["category"] for record in records)
    print(f"{name}（共 {len(records)} 条）")
    for category in CATEGORIES:
        count = counts[category]
        print(f"  {category}：{count} 条，占 {count / len(records):.0%}")


def main():
    # Step1、构造 80 条常见类和 20 条少见类记录，作为已知总体。
    population = [
        {"id": index, "category": "常见类" if index < 80 else "少见类"}
        for index in range(100)
    ]

    # Step2、从同一总体中选取同样大小的样本，比较两种抽样方式。
    # T1、简单随机抽样：每条记录被选中的机会相同。
    random_sample = Random(42).sample(population, SAMPLE_SIZE)

    # T2、分层抽样：先按类别分层，再按总体类别比例从各层随机抽取。
    strata = {
        category: [
            record for record in population if record["category"] == category
        ]
        for category in CATEGORIES
    }
    # 本例的 80%/20% 与样本量 20 使两层配额恰好都是整数。
    quotas = {
        category: SAMPLE_SIZE * len(records) // len(population)
        for category, records in strata.items()
    }
    generator = Random(42)
    stratified_sample = [
        record
        for category in CATEGORIES
        for record in generator.sample(strata[category], quotas[category])
    ]

    # Step3、对照类别比例：一次随机结果可能接近也可能偏离总体。
    show_distribution("总体", population)
    show_distribution("简单随机样本", random_sample)
    show_distribution("分层样本", stratified_sample)
    print("分层抽样只在已知类别比例上与总体一致，不保证其他特征也一致。")


if __name__ == "__main__":
    main()
