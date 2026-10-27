# -*- coding: utf-8 -*-
# 文件用途：对比 sklearn、PyTorch、TensorFlow 的模型导出与重新加载推理。
# 对应章节/小节：第 6 章 6.6.3 模型导出并推理代码实战
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python 六、模型训练、评估与推理/6.6、模型导出并推理/6.6.3_model_export_inference.py --framework sklearn
# 备注：可改选 pytorch、tensorflow；PMML 仅导出，加载推理需兼容的 PMML 运行时。

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    REPO_ROOT
    / "六、模型训练、评估与推理"
    / "6.6、模型导出并推理"
    / "outputs"
)


def make_demo_data():
    # Step1、准备训练数据及模拟的新样本；推理函数只接收新样本特征。
    features, labels = make_classification(
        n_samples=300, n_features=8, n_informative=5,
        n_redundant=0, random_state=42,
    )
    train_features, new_features, train_labels, _ = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    return train_features, new_features, train_labels


def save_scaler(scaler, path):
    """保存训练阶段拟合的标准化器。"""
    with path.open("wb") as stream:
        pickle.dump(scaler, stream)


def load_scaler(path):
    """重新载入本脚本保存的标准化器，用于新样本推理。"""
    with path.open("rb") as stream:
        return pickle.load(stream)


def sklearn_example(train_features, new_features, train_labels, export_pmml):
    # Step2、训练 sklearn Pipeline，让标准化参数也随模型保存。
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(train_features, train_labels)

    # T1、pickle 保存与加载完整 Pipeline，再对无标签样本推理。
    model_path = OUTPUT_DIR / "model_sklearn.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream)
    with model_path.open("rb") as stream:
        restored = pickle.load(stream)
    expected = model.predict_proba(new_features)[:, 1]
    actual = restored.predict_proba(new_features)[:, 1]
    np.testing.assert_allclose(actual, expected)
    print(f"sklearn 加载后前五条预测概率：{actual[:5]}")

    # T2、PMML 导出需要 sklearn2pmml 和兼容的 JDK；推理需另配 PMML 运行时。
    if export_pmml:
        from sklearn2pmml import PMMLPipeline, sklearn2pmml

        pmml_model = PMMLPipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=500)),
            ]
        )
        pmml_model.fit(train_features, train_labels)
        pmml_path = OUTPUT_DIR / "model_sklearn.pmml"
        sklearn2pmml(pmml_model, str(pmml_path), with_repr=True)
        print(f"PMML 已导出：{pmml_path}；加载推理需兼容的 PMML 运行时。")


def build_pytorch_model(feature_count):
    from torch import nn

    return nn.Sequential(
        nn.Linear(feature_count, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def pytorch_example(train_features, new_features, train_labels, export_onnx):
    import torch
    from torch import nn

    # Step2、用训练数据拟合缩放器和一个小型二分类网络。
    scaler = StandardScaler().fit(train_features)
    scaler_path = OUTPUT_DIR / "scaler_pytorch.pkl"
    save_scaler(scaler, scaler_path)
    train_array = scaler.transform(train_features).astype(np.float32)
    reference_array = scaler.transform(new_features).astype(np.float32)
    inputs = torch.from_numpy(train_array)
    targets = torch.from_numpy(train_labels.astype(np.float32)).unsqueeze(1)
    torch.manual_seed(42)
    model = build_pytorch_model(train_array.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
    model.eval()

    # T1、保存 state_dict，再用同样结构重建模型进行推理。
    weights_path = OUTPUT_DIR / "model_pytorch.pth"
    torch.save(model.state_dict(), weights_path)
    restored = build_pytorch_model(train_array.shape[1])
    restored.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    restored.eval()
    restored_scaler = load_scaler(scaler_path)
    new_array = restored_scaler.transform(new_features).astype(np.float32)
    with torch.no_grad():
        expected = torch.sigmoid(
            model(torch.from_numpy(reference_array))
        ).numpy().ravel()
        actual = torch.sigmoid(
            restored(torch.from_numpy(new_array))
        ).numpy().ravel()
    np.testing.assert_allclose(actual, expected)
    print(f"PyTorch 加载后前五条预测概率：{actual[:5]}")

    # T2、可选导出 ONNX，并让 ONNX Runtime 对相同的新样本推理。
    if export_onnx:
        import onnxruntime

        onnx_path = OUTPUT_DIR / "model_pytorch.onnx"
        torch.onnx.export(
            model, torch.zeros(1, train_array.shape[1]), str(onnx_path),
            input_names=["features"], output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=13,
        )
        session = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        logits = session.run(
            None, {session.get_inputs()[0].name: new_array}
        )[0].ravel()
        onnx_probabilities = 1 / (1 + np.exp(-logits))
        np.testing.assert_allclose(onnx_probabilities, expected, atol=1e-5)
        print(f"PyTorch ONNX 前五条预测概率：{onnx_probabilities[:5]}")


def tensorflow_example(train_features, new_features, train_labels, export_onnx):
    import tensorflow as tf

    # Step2、构造与另两种框架同目标的轻量二分类网络。
    scaler = StandardScaler().fit(train_features)
    scaler_path = OUTPUT_DIR / "scaler_tensorflow.pkl"
    save_scaler(scaler, scaler_path)
    train_array = scaler.transform(train_features).astype(np.float32)
    reference_array = scaler.transform(new_features).astype(np.float32)
    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(train_array.shape[1],), name="features"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(train_array, train_labels, epochs=20, batch_size=32, verbose=0)
    expected = model.predict(reference_array, verbose=0).ravel()

    # T1、H5 保存完整模型；重新加载后推理同一批无标签样本。
    h5_path = OUTPUT_DIR / "model_tensorflow.h5"
    model.save(h5_path)
    restored = tf.keras.models.load_model(h5_path, compile=False)
    restored_scaler = load_scaler(scaler_path)
    new_array = restored_scaler.transform(new_features).astype(np.float32)
    actual = restored.predict(new_array, verbose=0).ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    print(f"TensorFlow H5 前五条预测概率：{actual[:5]}")

    # T2、只保存权重时必须先构造同样的网络结构。
    weights_path = OUTPUT_DIR / "model_tensorflow.weights.h5"
    model.save_weights(weights_path)

    # T3、JSON 保存结构，配合权重文件才能恢复可推理模型。
    structure_path = OUTPUT_DIR / "model_tensorflow.json"
    structure_path.write_text(model.to_json(), encoding="utf-8")
    from_json = tf.keras.models.model_from_json(
        structure_path.read_text(encoding="utf-8")
    )
    from_json.load_weights(weights_path)
    json_prediction = from_json.predict(new_array, verbose=0).ravel()
    np.testing.assert_allclose(json_prediction, expected, atol=1e-5)
    print(f"TensorFlow JSON 加权重后前五条预测概率：{json_prediction[:5]}")

    # T4、可选转成 ONNX，让独立推理引擎读取模型。
    if export_onnx:
        import onnxruntime
        import tf2onnx

        onnx_path = OUTPUT_DIR / "model_tensorflow.onnx"
        tf2onnx.convert.from_keras(
            model,
            input_signature=(
                tf.TensorSpec(
                    [None, train_array.shape[1]], tf.float32, name="features"
                ),
            ),
            output_path=str(onnx_path),
        )
        session = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        onnx_prediction = session.run(
            None, {session.get_inputs()[0].name: new_array}
        )[0].ravel()
        np.testing.assert_allclose(onnx_prediction, expected, atol=1e-5)
        print(f"TensorFlow ONNX 前五条预测概率：{onnx_prediction[:5]}")


def main():
    parser = argparse.ArgumentParser(description="模型导出和无标签样本推理")
    parser.add_argument(
        "--framework", choices=["sklearn", "pytorch", "tensorflow"],
        default="sklearn",
    )
    parser.add_argument("--export-pmml", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    arguments = parser.parse_args()
    if arguments.export_pmml and arguments.framework != "sklearn":
        parser.error("--export-pmml 仅适用于 sklearn。")
    if arguments.export_onnx and arguments.framework == "sklearn":
        parser.error("--export-onnx 仅适用于 pytorch 或 tensorflow。")

    train_features, new_features, train_labels = make_demo_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step3、按所选框架导出，再验证重新加载后的推理结果。
    if arguments.framework == "sklearn":
        sklearn_example(
            train_features, new_features, train_labels,
            arguments.export_pmml,
        )
    elif arguments.framework == "pytorch":
        pytorch_example(
            train_features, new_features, train_labels,
            arguments.export_onnx,
        )
    else:
        tensorflow_example(
            train_features, new_features, train_labels,
            arguments.export_onnx,
        )


if __name__ == "__main__":
    main()
