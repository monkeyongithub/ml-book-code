# -*- coding: utf-8 -*-
# 文件用途：向本机预测 API 发送多条数值化的 Titanic 样本。
# 对应章节/小节：第 7 章 7.3.4 模型部署代码实战
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：先从仓库根目录启动预测服务，再执行 python "七、模型发布、部署与监控/7.3、模型部署/7.3.4_batch_prediction_request.py"。
# 备注：批量接口与单条接口使用相同的特征字段。

import requests

API_URL = "http://127.0.0.1:5000/predict_batch"


def main():
    # Step1、准备与单条接口结构一致的两条样本。
    samples = [
        {
            "pclass": 1.0,
            "age": 49.0,
            "sibsp": 1.0,
            "fare": 56.9292,
            "sex_female": 0.0,
            "sex_male": 1.0,
            "embarked_C": 1.0,
            "embarked_Q": 0.0,
            "embarked_S": 0.0,
        },
        {
            "pclass": 3.0,
            "age": 25.0,
            "sibsp": 0.0,
            "fare": 7.925,
            "sex_female": 1.0,
            "sex_male": 0.0,
            "embarked_C": 0.0,
            "embarked_Q": 0.0,
            "embarked_S": 1.0,
        },
    ]

    # Step2、向本机服务的批量预测接口发送 POST 请求。
    try:
        response = requests.post(API_URL, json=samples, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit("请求失败，请先启动预测 API 并检查请求字段。") from error

    # Step3、按输入样本的顺序展示 predictions。
    for index, prediction in enumerate(response.json()["predictions"], start=1):
        print(f"第 {index} 条预测结果：{prediction}")


if __name__ == "__main__":
    main()
