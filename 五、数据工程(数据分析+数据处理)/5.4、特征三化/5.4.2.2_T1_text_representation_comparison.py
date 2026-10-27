# -*- coding: utf-8 -*-
# 文件用途：对照书稿比较同一单词的词袋、TF-IDF 与 Word2Vec 表示。
# 对应章节/小节：第 5 章 5.4.2.2 广义的文本型数据编码
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.4、特征三化/5.4.2.2_T1_text_representation_comparison.py"
# 备注：T1 对应书稿中的单词表示对照示例。

import re

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


DOCUMENTS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
    "The last document is here.",
]
COMPARE_WORD = "document"


def main():
    # Step1、用书稿中的同一组文本构造词袋矩阵和 TF-IDF 矩阵。
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(DOCUMENTS)
    tfidf_vectorizer = TfidfVectorizer()
    weights = tfidf_vectorizer.fit_transform(DOCUMENTS)

    # Step2、分词后训练小型 Word2Vec，取得“单词”本身的向量。
    tokenized = [re.findall(r"[a-z]+", doc.lower()) for doc in DOCUMENTS]
    word2vec = Word2Vec(
        tokenized, vector_size=5, window=2, min_count=1, workers=1, seed=42
    )

    # T1、同词表示对照：比较词袋计数、TF-IDF 权重和 Word2Vec 向量。
    # Step3、对同一个单词逐项对照；前两项每篇文档一个值，词向量每词一个向量。
    count_index = count_vectorizer.vocabulary_[COMPARE_WORD]
    tfidf_index = tfidf_vectorizer.vocabulary_[COMPARE_WORD]
    print("单词：", COMPARE_WORD)
    print("词袋计数：", counts[:, count_index].toarray().ravel())
    print("TF-IDF 权重：", weights[:, tfidf_index].toarray().ravel())
    print("Word2Vec 词向量：", word2vec.wv[COMPARE_WORD])


if __name__ == "__main__":
    main()
