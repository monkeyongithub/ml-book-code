# -*- coding: utf-8 -*-
# 文件用途：以 MovieLens 用户与电影关系演示图特征构造。
# 对应章节/小节：第 5 章 5.3.2 基于纯技术构造特征
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.3、构造特征/5.3.2_graph_features.py"
# 备注：读取本章附带的评分 CSV 重建二部图；不下载外部数据。

from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = (
    REPO_ROOT
    / "五、数据工程(数据分析+数据处理)/degree_centrality_clustering_coefficient.csv"
)
FEATURE_COLUMNS = ("user_degree", "user_clustering", "movie_degree")


def add_graph_features(ratings, degree_centrality, user_clustering):
    # 按用户和电影节点为每条评分记录补充图特征。
    result = ratings.copy()
    result["user_degree"] = result["userId"].map(
        lambda user_id: degree_centrality.get(("user", int(user_id)), 0.0)
    )
    result["user_clustering"] = result["userId"].map(
        lambda user_id: user_clustering.get(("user", int(user_id)), 0.0)
    )
    result["movie_degree"] = result["movieId"].map(
        lambda movie_id: degree_centrality.get(("movie", int(movie_id)), 0.0)
    )
    return result


def main():
    # T7、基于用户—电影二部图的连接关系构造图特征。
    # Step1、从配书数据中读取原始评分列，抽取小样本便于观察图结构。
    ratings = pd.read_csv(
        RATINGS_PATH, usecols=["userId", "movieId", "rating"]
    ).sample(n=2000, random_state=42)
    train_rows, test_rows = train_test_split(
        ratings, test_size=0.2, random_state=42
    )

    # Step2、只用训练记录建二部图；不同类型节点用前缀避免 ID 碰撞。
    graph = nx.Graph()
    for row in train_rows.itertuples(index=False):
        graph.add_edge(("user", int(row.userId)), ("movie", int(row.movieId)))
    user_nodes = [node for node in graph if node[0] == "user"]
    degree_centrality = nx.degree_centrality(graph)
    user_clustering = bipartite.clustering(graph, nodes=user_nodes)

    # Step3、将中心性和二部图聚类系数融合为特征，并演示评分回归。
    train_features = add_graph_features(
        train_rows, degree_centrality, user_clustering
    )
    test_features = add_graph_features(
        test_rows, degree_centrality, user_clustering
    )
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(train_features[list(FEATURE_COLUMNS)], train_features["rating"])
    predictions = model.predict(test_features[list(FEATURE_COLUMNS)])
    print(train_features[list(FEATURE_COLUMNS)].head())
    print(f"评分回归 MSE：{mean_squared_error(test_features['rating'], predictions):.4f}")


if __name__ == "__main__":
    main()
