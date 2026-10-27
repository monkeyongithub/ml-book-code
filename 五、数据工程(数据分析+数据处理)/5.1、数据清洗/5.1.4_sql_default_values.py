# -*- coding: utf-8 -*-
# 文件用途：演示 SQL 字段的 DEFAULT 值在插入数据时何时生效。
# 对应章节/小节：第 5 章 5.1.4.2 特殊值的分析与处理：代码实战
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 python "五、数据工程(数据分析+数据处理)/5.1、数据清洗/5.1.4_sql_default_values.py"
# 备注：使用 SQLite 内存数据库；Python 的 None 对应 SQL NULL。

from contextlib import closing
import sqlite3


def main():
    # Step1、参照书稿的整数字段示例，创建默认年龄为 18 的内存表。
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            """
            CREATE TABLE ExampleTable (
                ID INTEGER PRIMARY KEY,
                Age INTEGER DEFAULT 18
            )
            """
        )

        # Step2、分别插入省略字段、显式写入 0、显式写入 NULL 的记录。
        connection.execute("INSERT INTO ExampleTable (ID) VALUES (?)", (1,))
        connection.execute(
            "INSERT INTO ExampleTable (ID, Age) VALUES (?, ?)",
            (2, 0),
        )
        connection.execute(
            "INSERT INTO ExampleTable (ID, Age) VALUES (?, ?)",
            (3, None),
        )

        # Step3、对照结果：只有省略字段的记录使用 DEFAULT；0 和 NULL 保留原值。
        for row in connection.execute(
            "SELECT ID, Age FROM ExampleTable ORDER BY ID"
        ):
            print(f"ID={row[0]}，Age={row[1]!r}")


if __name__ == "__main__":
    main()
