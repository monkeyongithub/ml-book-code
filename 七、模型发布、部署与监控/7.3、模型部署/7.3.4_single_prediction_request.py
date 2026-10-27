# -*- coding: utf-8 -*-
# 文件用途：向本机预测 API 发送一条数值化的 Titanic 样本。
# 对应章节/小节：第 7 章 7.3.4 模型部署代码实战
# 作者：NYY
# 创建日期：2024-03-20
# 运行方式：先从仓库根目录启动预测服务，再执行 python "七、模型发布、部署与监控/7.3、模型部署/7.3.4_single_prediction_request.py"。
# 备注：请求字段与服务端训练模型使用的特征列保持一致。

import requests

API_URL = "http://127.0.0.1:5000/predict"


def main():
    # Step1、按书稿的编码后特征格式准备单条样本。
    sample = {
        "pclass": 1.0,
        "age": 49.0,
        "sibsp": 1.0,
        "fare": 56.9292,
        "sex_female": 0.0,
        "sex_male": 1.0,
        "embarked_C": 1.0,
        "embarked_Q": 0.0,
        "embarked_S": 0.0,
    }

    # Step2、向本机服务的单条预测接口发送 POST 请求。
    try:
        response = requests.post(API_URL, json=sample, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit("请求失败，请先启动预测 API 并检查请求字段。") from error

    # Step3、读取与服务端约定的 prediction 字段。
    print(f"预测结果：{response.json()['prediction']}")


if __name__ == "__main__":
    main()
