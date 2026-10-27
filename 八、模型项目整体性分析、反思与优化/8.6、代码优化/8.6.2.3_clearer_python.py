# -*- coding: utf-8 -*-
# 文件用途：比较列表与字典推导式，以及 enumerate 对遍历代码的简化。
# 对应章节/小节：第 8 章 8.6.2.3 代码简洁优化示例
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "八、模型项目整体性分析、反思与优化/8.6、代码优化/8.6.2.3_clearer_python.py"。
# 备注：相同输入才适合比较执行时间；代码简洁性主要体现在可读性。

from time import perf_counter


def main():
    # Step1、用相同输入比较列表构造的两种写法。
    values = range(10000)

    # T1、显式循环构造平方数列表。
    started = perf_counter()
    squares_loop = []
    for value in values:
        squares_loop.append(value ** 2)
    loop_seconds = perf_counter() - started
    # T2、列表推导式构造相同的平方数。
    started = perf_counter()
    squares_comp = [value ** 2 for value in values]
    comp_seconds = perf_counter() - started
    assert squares_loop == squares_comp
    print(f"列表循环：{loop_seconds:.6f} 秒；推导式：{comp_seconds:.6f} 秒")

    # Step2、用相同输入比较字典构造的两种写法。
    # T1、显式循环构造映射。
    started = perf_counter()
    mapping_loop = {}
    for value in values:
        mapping_loop[value] = value ** 2
    loop_seconds = perf_counter() - started
    # T2、字典推导式构造相同的映射。
    started = perf_counter()
    mapping_comp = {value: value ** 2 for value in values}
    comp_seconds = perf_counter() - started
    assert mapping_loop == mapping_comp
    print(f"字典循环：{loop_seconds:.6f} 秒；推导式：{comp_seconds:.6f} 秒")

    # Step3、用 enumerate 同时取得索引与元素，无需手动维护计数器。
    for index, value in enumerate(["训练", "验证", "测试"]):
        print(index, value)


if __name__ == "__main__":
    main()
