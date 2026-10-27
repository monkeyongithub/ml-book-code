# -*- coding: utf-8 -*-
# 文件用途：扩展示范 Word2Vec、Doc2Vec 与预训练句向量。
# 对应章节/小节：第 5 章 5.4.2.2 广义的文本型数据编码
# 作者：NYY
# 创建日期：2024-11-30
# 运行方式：从仓库根目录执行 PYTHONPATH=. python "五、数据工程(数据分析+数据处理)/5.4、特征三化/5.4.2.2_T2_text_embedding_methods.py"
# 备注：T2 展示不同粒度的嵌入；预训练模型可能需要下载。

import re

from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument


DOCUMENTS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
RUN_PRETRAINED_EXAMPLE = False


def main():
    # Step1、沿用同一批文本，先得到单词序列。
    tokenized = [re.findall(r"[a-z]+", doc.lower()) for doc in DOCUMENTS]

    # T2、不同粒度的嵌入：依次查看词向量、文档向量与按需加载的句向量。
    # Step2、Word2Vec 为单词建向量；Doc2Vec 为整篇文档建向量。
    word_model = Word2Vec(
        tokenized, vector_size=20, window=2, min_count=1, workers=1, seed=42
    )
    tagged = [
        TaggedDocument(words=tokens, tags=[str(index)])
        for index, tokens in enumerate(tokenized)
    ]
    document_model = Doc2Vec(
        tagged, vector_size=20, window=2, min_count=1, workers=1, seed=42
    )
    print("单词 document 的向量形状：", word_model.wv["document"].shape)
    print("第 0 篇文档的向量形状：", document_model.dv["0"].shape)

    # Step3、预训练模型可直接生成句向量；首次加载时可能下载权重。
    if RUN_PRETRAINED_EXAMPLE:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        sentence_vectors = model.encode(DOCUMENTS, normalize_embeddings=True)
        print("预训练句向量矩阵形状：", sentence_vectors.shape)


if __name__ == "__main__":
    main()
