# -*- coding: utf-8 -*-
# 文件用途：演示自动编码器、CNN、RNN 与预训练模型提取特征。
# 对应章节/小节：第 5 章 5.3.4 利用深度学习技术自动构造特征
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.3、构造特征/5.3.4_deep_feature_extraction.py"
# 备注：书稿的 T3 GAN 属于概念介绍，本示例未训练 GAN；数据集与预训练模型首次使用时可能需要下载。

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


VOCAB_SIZE = 1000
SEQUENCE_LENGTH = 100
RUN_PRETRAINED_EXAMPLE = False


def multi_hot(sequences, vocabulary_size):
    # 把词 ID 序列转为适合二值重建的多热向量。
    matrix = np.zeros((len(sequences), vocabulary_size), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        matrix[row, np.asarray(sequence, dtype=int)] = 1.0
    return matrix


def main():
    # Step1、载入 IMDb 评论；少量样本足够演示四种特征提取思路。
    (train_sequences, y_train), (test_sequences, y_test) = imdb.load_data(
        num_words=VOCAB_SIZE
    )
    train_sequences, y_train = train_sequences[:2000], y_train[:2000]
    test_sequences, y_test = test_sequences[:500], y_test[:500]

    # T1、自动编码器：重建多热词袋，并取中间层作为文档特征。
    train_bow = multi_hot(train_sequences, VOCAB_SIZE)
    test_bow = multi_hot(test_sequences, VOCAB_SIZE)
    bow_input = Input(shape=(VOCAB_SIZE,))
    encoded = Dense(64, activation="relu")(bow_input)
    reconstructed = Dense(VOCAB_SIZE, activation="sigmoid")(encoded)
    autoencoder = Model(bow_input, reconstructed)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(train_bow, train_bow, epochs=2, batch_size=64, verbose=0)
    encoder = Model(bow_input, encoded)
    print("T1 编码特征形状：", encoder.predict(test_bow, verbose=0).shape)
    print("T1 重建损失：", autoencoder.evaluate(test_bow, test_bow, verbose=0))

    # Step2、CNN 和 RNN 使用相同的填充序列，便于比较提取的表示。
    x_train = pad_sequences(train_sequences, maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH)

    # T2、CNN 与 RNN 共用相同的序列输入，分别提取评论特征。
    # CNN：池化层输出固定长度的评论特征。
    cnn_input = Input(shape=(SEQUENCE_LENGTH,), dtype="int32")
    cnn_embedded = Embedding(VOCAB_SIZE, 32)(cnn_input)
    convolution = Conv1D(64, 5, activation="relu")(cnn_embedded)
    cnn_features = GlobalMaxPooling1D()(convolution)
    cnn_model = Model(cnn_input, Dense(1, activation="sigmoid")(cnn_features))
    cnn_model.compile(optimizer="adam", loss="binary_crossentropy")
    cnn_model.fit(x_train, y_train, epochs=2, batch_size=64, verbose=0)
    cnn_encoder = Model(cnn_input, cnn_features)
    print("T2 CNN 特征形状：", cnn_encoder.predict(x_test, verbose=0).shape)

    # RNN：LSTM 最后一层状态是固定长度的评论特征。
    rnn_input = Input(shape=(SEQUENCE_LENGTH,), dtype="int32")
    rnn_embedded = Embedding(VOCAB_SIZE, 32)(rnn_input)
    rnn_features = LSTM(64)(rnn_embedded)
    rnn_model = Model(rnn_input, Dense(1, activation="sigmoid")(rnn_features))
    rnn_model.compile(optimizer="adam", loss="binary_crossentropy")
    rnn_model.fit(x_train, y_train, epochs=2, batch_size=64, verbose=0)
    rnn_encoder = Model(rnn_input, rnn_features)
    print("T2 RNN 特征形状：", rnn_encoder.predict(x_test, verbose=0).shape)

    # Step3、按需查看预训练模型的句向量，与前面三种特征粒度比较。
    # T4、迁移学习：用预训练模型把可读文本转换为句向量。
    if RUN_PRETRAINED_EXAMPLE:
        from sentence_transformers import SentenceTransformer

        sentences = ["This movie is excellent.", "The plot is dull."]
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        embeddings = model.encode(sentences, normalize_embeddings=True)
        print("T4 预训练句向量形状：", embeddings.shape)


if __name__ == "__main__":
    main()
