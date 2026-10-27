# -*- coding: utf-8 -*-
# 文件用途：模拟记录用户行为事件，理解埋点收集的数据结构。
# 对应章节/小节：第 3 章 3.2.2.2 数据渠道代码实战
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 三、数据科学任务完整流程/3.2、数据认知与数据收集/3.2.2、数据渠道/3.2.2.2_analytics_tracking.py
# 备注：事件与用户均为模拟数据，不连接真实网站或数据服务。

from datetime import datetime, timezone


class AnalyticsTracker:
    """用事件名、用户编号和时间戳记录最小埋点数据。"""

    def __init__(self):
        self._events = []

    def track_event(self, event_name, user_id, timestamp=None):
        if not event_name or user_id is None:
            raise ValueError("事件名和用户编号不能为空。")
        event_time = (
            timestamp if timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        )
        event = {
            "event_name": event_name,
            "user_id": user_id,
            "timestamp": event_time,
        }
        self._events.append(event)

    def get_data(self):
        """返回记录的副本，避免外部代码改写内部列表。"""
        return list(self._events)


def simulate_user_behavior():
    # Step1、创建一个简单的事件记录器。
    tracker = AnalyticsTracker()

    # Step2、按浏览、加购、完成购买的顺序模拟两个用户。
    for event_name in ("view_product", "add_to_cart", "purchase_complete"):
        for user_id in (1, 2):
            tracker.track_event(event_name, user_id)
    return tracker.get_data()


def main():
    # Step3、观察每条事件包含的用户、行为与时间信息。
    for event in simulate_user_behavior():
        print(event)


if __name__ == "__main__":
    main()
