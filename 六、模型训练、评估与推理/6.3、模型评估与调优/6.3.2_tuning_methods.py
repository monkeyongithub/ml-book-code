# -*- coding: utf-8 -*-
# 文件用途：用同一决策树示例比较手动、随机、贪心、网格和贝叶斯调参。
# 对应章节/小节：第 6 章 6.3.2.2 模型调参方法
# 作者：NYY
# 创建日期：2024-03-06
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.3、模型评估与调优/6.3.2_tuning_methods.py
# 备注：贝叶斯调参示例需要 hyperopt；默认关闭，可按需启用。

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier


PARAMETER_GRID = {
    "max_depth": [2, 4, 6],
    "min_samples_leaf": [1, 3],
}


def evaluate_parameters(parameters, features, labels, folds):
    model = DecisionTreeClassifier(random_state=42, **parameters)
    return cross_val_score(
        model, features, labels, cv=folds, scoring="accuracy"
    ).mean()


def manual_search(features, labels, folds):
    # T1、手动调参：先基于经验列出少量明确候选。
    candidates = [
        {"max_depth": 2, "min_samples_leaf": 1},
        {"max_depth": 4, "min_samples_leaf": 3},
    ]
    return max(
        candidates,
        key=lambda params: evaluate_parameters(params, features, labels, folds),
    )


def random_search(features, labels, folds):
    # T2、随机调参：从给定空间抽取有限个参数组合。
    search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        PARAMETER_GRID, n_iter=4, cv=folds, scoring="accuracy",
        random_state=42,
    )
    search.fit(features, labels)
    return search.best_params_


def greedy_search(features, labels, folds):
    # T3、贪心调参：逐个参数尝试，仅保留当前改进。
    best = {"max_depth": 2, "min_samples_leaf": 1}
    best_score = evaluate_parameters(best, features, labels, folds)
    for parameter, values in PARAMETER_GRID.items():
        for value in values:
            candidate = {**best, parameter: value}
            score = evaluate_parameters(candidate, features, labels, folds)
            if score > best_score:
                best, best_score = candidate, score
    return best


def grid_search(features, labels, folds):
    # T4、网格调参：遍历所有离散组合。
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        PARAMETER_GRID, cv=folds, scoring="accuracy",
    )
    search.fit(features, labels)
    return search.best_params_


def bayesian_search(features, labels, folds):
    # T5、贝叶斯调参：根据已尝试组合的结果选择下一组。
    from hyperopt import Trials, fmin, hp, space_eval, tpe

    space = {
        "max_depth": hp.choice("max_depth", PARAMETER_GRID["max_depth"]),
        "min_samples_leaf": hp.choice(
            "min_samples_leaf", PARAMETER_GRID["min_samples_leaf"]
        ),
    }
    best_indices = fmin(
        fn=lambda params: -evaluate_parameters(
            params, features, labels, folds
        ),
        space=space, algo=tpe.suggest, max_evals=6,
        trials=Trials(), rstate=None, show_progressbar=False,
    )
    return space_eval(space, best_indices)


def main():
    # Step1、建立同一数据、模型和分层交叉验证规则，保证方法可比。
    features, labels = make_classification(
        n_samples=200, n_features=8, n_informative=5,
        n_redundant=0, random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Step2、在训练集内比较调参策略，不让测试集参与挑选。
    strategies = {
        "手动": manual_search,
        "随机": random_search,
        "贪心": greedy_search,
        "网格": grid_search,
    }
    run_bayesian = False  # 如已准备 hyperopt，可改为 True 比较 T5。
    if run_bayesian:
        strategies["贝叶斯"] = bayesian_search
    for name, strategy in strategies.items():
        best = strategy(x_train, y_train, folds)
        print(f"{name}调参的候选参数：{best}")

    # Step3、选定策略后，再用独立测试集做最终评估。
    selected = grid_search(x_train, y_train, folds)
    final_model = DecisionTreeClassifier(random_state=42, **selected)
    final_model.fit(x_train, y_train)
    print(f"网格策略的测试准确率：{final_model.score(x_test, y_test):.3f}")


if __name__ == "__main__":
    main()
