"""
GloVe 词嵌入模型演示 — glove.6B.50d

GloVe（Global Vectors for Word Representation）是 Stanford 提出的静态词嵌入方法。
与 Word2Vec 仅利用局部上下文窗口不同，GloVe 利用全局词共现矩阵的统计信息来学习词向量。

核心思想：词向量之间的偏移量（向量差）应能捕捉词之间的语义关系。
  例如：vector("king") - vector("man") + vector("woman") ≈ vector("queen")

模型配置：
  - 训练数据：Wikipedia 2014 + Gigaword 5（约 60 亿 tokens）
  - 词表大小：400,000 词
  - 向量维度：50
  - 算法：加权最小二乘回归（Weighted Least Squares Regression）

本示例：
  1. 自动下载并加载预训练 GloVe 词向量（本地缓存）
  2. 查看词向量的数值表示和统计信息
  3. 计算词间余弦相似度
  4. 查找语义最近邻（most similar words）
  5. 词类比推理（word analogy）
  6. PCA 降维可视化语义空间
  7. 语义关系对比分析

首次运行会自动下载 glove.6B.zip（约 862MB）并提取 glove.6B.50d.txt（约 168MB）。
解压后 zip 包会被自动删除以节省空间。
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# ── 确保 stdout 使用 UTF-8（解决 Windows GBK 编码问题） ────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # 无头模式，避免 GUI 弹窗
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# ── 常量 ─────────────────────────────────────────────────────
EMBEDDING_DIM = 50
MAX_VOCAB = 400_000                       # GloVe 6B 50d 词表大小

# 下载源（自动处理重定向）
GLOVE_ZIP_URLS = [
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
]

# 本地缓存路径
CACHE_DIR = Path.home() / ".cache" / "glove_demo"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "glove.6B.50d.txt"
ZIP_CACHE = CACHE_DIR / "glove.6B.zip"


# ─────────────────────────────────────────────────────────────
# 1. 下载与加载
# ─────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, message: str = "") -> bool:
    """
    流式下载文件到本地，显示进度条。
    返回 True 表示下载成功。
    """
    try:
        if message:
            print(f"  {message}")
        print(f"  目标 URL：{url}")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        last_logged_pct = 0

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        # 每 10% 或每 50MB 输出一次进度
                        if pct - last_logged_pct >= 10 or downloaded - last_logged_pct * total / 100 >= 50 * 1024 * 1024:
                            print(f"    ├ 进度：{downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB ({pct:.0f}%)")
                            last_logged_pct = pct

        size_mb = downloaded / 1024 / 1024
        print(f"    └ 完成！{dest.name}（{size_mb:.0f} MB）")
        return True
    except Exception as e:
        print(f"    ✗ 下载失败：{e}")
        return False


def _extract_from_zip(zip_path: Path, target_file: str, output_path: Path) -> bool:
    """
    从 zip 中提取指定文件到 output_path。
    """
    try:
        print(f"  从 {zip_path.name} 中解压 {target_file} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            if target_file not in zf.namelist():
                files = [f for f in zf.namelist() if f.endswith(".txt")]
                print(f"    找不到 {target_file}，可用文件：{files}")
                return False
            with zf.open(target_file) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"    └ 解压完成：{output_path.name}（{size_mb:.0f} MB）")
        return True
    except Exception as e:
        print(f"    ✗ 解压失败：{e}")
        return False


def ensure_glove_file() -> Path:
    """
    确保 GloVe 向量文件存在（如有缓存则跳过下载）。
    返回文件路径。
    """
    if CACHE_FILE.exists():
        size_mb = CACHE_FILE.stat().st_size / 1024 / 1024
        print(f"  ✓ 使用本地缓存：{CACHE_FILE}（{size_mb:.0f} MB）")
        return CACHE_FILE

    print("=" * 60)
    print("  首次运行：下载 GloVe 预训练词向量")
    print(f"  下载包：glove.6B.zip（约 862MB）")
    print(f"  提取目标：glove.6B.50d.txt（约 168MB）")
    print(f"  缓存路径：{CACHE_DIR}")
    print("=" * 60)
    print()

    # 如果已有部分下载的 zip，继续使用
    for url in GLOVE_ZIP_URLS:
        if ZIP_CACHE.exists():
            print(f"  使用已存在的 zip 缓存：{ZIP_CACHE}")
        elif _download_file(url, ZIP_CACHE, f"从 {url} 下载 glove.6B.zip ..."):
            pass
        else:
            continue

        if _extract_from_zip(ZIP_CACHE, "glove.6B.50d.txt", CACHE_FILE):
            # 解压成功后删除 zip
            zip_size = ZIP_CACHE.stat().st_size / 1024 / 1024
            ZIP_CACHE.unlink(missing_ok=True)
            print(f"  已删除 zip 缓存以节省空间（{zip_size:.0f} MB）")
            return CACHE_FILE
        else:
            # 解压失败则删除损坏的 zip 并尝试下一个源
            ZIP_CACHE.unlink(missing_ok=True)

    raise RuntimeError(
        "所有下载源均失败。请手动下载 glove.6B.zip 并解压出 glove.6B.50d.txt 放到：\n"
        f"  {CACHE_FILE}\n\n"
        "下载地址：\n"
        "  https://nlp.stanford.edu/projects/glove/\n"
        "  https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip\n"
    )


def load_glove(filepath: Path, max_vocab: int = MAX_VOCAB) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """
    加载 GloVe 向量文件。

    文件格式（每行）：
      word 0.123 0.456 ... 0.789

    返回：
      word_to_vec : {词 -> 向量}
      vectors     : 所有向量的堆叠 (N, D)
      words       : 词列表（与 vectors 行对应）
    """
    print(f"\n  加载词向量（最多 {max_vocab:,} 词）...")
    word_to_vec: dict[str, np.ndarray] = {}
    vectors_list: list[np.ndarray] = []
    words_list: list[str] = []

    loaded = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < EMBEDDING_DIM + 1:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:EMBEDDING_DIM + 1]], dtype=np.float32)
            word_to_vec[word] = vec
            words_list.append(word)
            vectors_list.append(vec)
            loaded += 1
            if loaded >= max_vocab:
                break

    vectors = np.stack(vectors_list)  # (N, D)

    print(f"  ✓ 加载完成：{len(word_to_vec):,} 个词 × {EMBEDDING_DIM} 维")
    print(f"  向量矩阵形状：{vectors.shape}")
    print(f"  数据类型：{vectors.dtype}")
    return word_to_vec, vectors, words_list


# ─────────────────────────────────────────────────────────────
# 2. 向量运算工具
# ─────────────────────────────────────────────────────────────

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 归一化向量矩阵（每行除以其范数）。"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return vectors / norms


def find_similar_words(
    query_vec: np.ndarray,
    word_to_vec: dict[str, np.ndarray],
    normalized: bool = False,
    topk: int = 10,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    """
    在词表中查找与 query_vec 最相似的 topk 个词。

    参数：
      normalized : query_vec 和词表是否已 L2 归一化
                   （已归一化时直接用点积替代余弦相似度，速度更快）
    """
    exclude = exclude or set()
    results: list[tuple[str, float]] = []

    for word, vec in word_to_vec.items():
        if word in exclude:
            continue
        if normalized:
            sim = float(np.dot(query_vec, vec))
        else:
            sim = cosine_similarity(query_vec, vec)
        results.append((word, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topk]


def word_analogy(
    a: str, b: str, c: str,
    word_to_vec: dict[str, np.ndarray],
    word_to_vec_norm: dict[str, np.ndarray] | None = None,
    topk: int = 10,
) -> list[tuple[str, float]]:
    """
    词类比推理：a 之于 b，如同 c 之于 ?
    即：vec(b) - vec(a) + vec(c) 的最近邻（排除 a, b, c 本身）。

    经典案例：
      king - man + woman = queen
      paris - france + italy = rome
    """
    va = word_to_vec.get(a)
    vb = word_to_vec.get(b)
    vc = word_to_vec.get(c)

    if va is None or vb is None or vc is None:
        missing = [w for w, v in [(a, va), (b, vb), (c, vc)] if v is None]
        print(f"    ✗ 词表中找不到：{', '.join(missing)}")
        return []

    target_vec = vb - va + vc
    exclude = {a, b, c}

    if word_to_vec_norm is not None and query_norm(target_vec) is not None:
        # 使用归一化词表进行点积搜索（更快）
        return find_similar_words(
            target_vec / (np.linalg.norm(target_vec) + 1e-10),
            word_to_vec_norm,
            normalized=True,
            topk=topk,
            exclude=exclude,
        )

    return find_similar_words(target_vec, word_to_vec, topk=topk, exclude=exclude)


def query_norm(vec: np.ndarray) -> np.ndarray | None:
    """将向量 L2 归一化。"""
    n = np.linalg.norm(vec)
    if n < 1e-10:
        return None
    return vec / n


# ─────────────────────────────────────────────────────────────
# 3. PCA 可视化（纯 numpy 实现）
# ─────────────────────────────────────────────────────────────

def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """
    用 SVD 实现 PCA 降维到 2D。
    不需要 sklearn，纯 numpy 实现。
    """
    # 中心化
    mean = vectors.mean(axis=0)
    centered = vectors - mean

    # SVD 分解
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # 投影到前两个主成分
    projected = centered @ Vt[:2].T
    return projected


# ─────────────────────────────────────────────────────────────
# 4. 演示函数
# ─────────────────────────────────────────────────────────────

def demo_basic_info(word_to_vec: dict[str, np.ndarray]) -> None:
    """展示 GloVe 模型基本信息。"""
    print()
    print("=" * 70)
    print("  [基本信息] GloVe 6B 50d 词向量")
    print("=" * 70)
    print(f"  ├ 词表大小：{len(word_to_vec):,} 词")
    print(f"  ├ 向量维度：{EMBEDDING_DIM} 维")
    print(f"  ├ 训练数据：Wikipedia 2014 + Gigaword 5")
    print(f"  ├ 训练 tokens：约 60 亿")
    print(f"  ├ 算法：GloVe（加权共现矩阵分解）")
    print(f"  └ 文件大小：约 168 MB（文本格式）")

    # 取一个样本词查看向量
    sample_words = ["king", "queen", "apple", "computer", "china"]
    print()
    print(f"  样本词向量快照：")
    print(f"  {'词':<12} {'前 8 维值':<50} {'L2范数':<10}")
    print(f"  {'-' * 74}")
    for w in sample_words:
        if w in word_to_vec:
            vec = word_to_vec[w]
            front = ", ".join(f"{v:+.4f}" for v in vec[:8])
            norm = np.linalg.norm(vec)
            print(f"  {w:<12} [{front}, ...]  {norm:<10.4f}")
    print()


def demo_word_embeddings(word_to_vec: dict[str, np.ndarray]) -> None:
    """展示指定词的嵌入向量详细信息。"""
    print()
    print("=" * 70)
    print("  [词向量详解] 查看任意词的嵌入表示")
    print("=" * 70)

    target_words = ["king", "queen", "deep", "learning", "computer"]

    for word in target_words:
        if word not in word_to_vec:
            continue
        vec = word_to_vec[word]
        print(f"\n  ── {word} ──")
        print(f"    维度：{len(vec)}")
        print(f"    向量值（全部 50 维）：")

        # 分多行显示
        for row in range(5):
            start = row * 10
            end = start + 10
            vals = ", ".join(f"{v:+.4f}" for v in vec[start:end])
            print(f"      [{row * 10:2d}-{end:2d}]  {vals}")

        # 统计信息
        print(f"    统计：均值={vec.mean():+.6f}, 标准差={vec.std():.6f}, "
              f"L2范数={np.linalg.norm(vec):.4f}, "
              f"最小值={vec.min():+.6f}, 最大值={vec.max():+.6f}")
    print()


def demo_similarity(word_to_vec: dict[str, np.ndarray]) -> None:
    """展示词间相似度计算。"""
    print()
    print("=" * 70)
    print("  [语义相似度] 词对余弦相似度对比")
    print("=" * 70)
    print(f"  语义相近的词 → 高相似度（接近 1.0）")
    print(f"  语义无关的词 → 低相似度（接近 0.0）")
    print(f"  语义相反的词 → 负相似度（接近 -1.0）")
    print()

    # 近义词对
    similar_pairs = [
        ("king", "queen"),
        ("car", "automobile"),
        ("happy", "glad"),
        ("computer", "laptop"),
        ("ocean", "sea"),
        ("dog", "cat"),
        ("apple", "banana"),
        ("fast", "quick"),
    ]

    # 不相关词对
    unrelated_pairs = [
        ("king", "apple"),
        ("computer", "ocean"),
        ("dog", "mathematics"),
    ]

    # 反义/对比词对
    antonym_pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("good", "bad"),
        ("love", "hate"),
        ("up", "down"),
    ]

    print("  ▸ 语义相近词对：")
    print(f"  {'词1':<12} {'词2':<12} {'余弦相似度':<14} {'说明'}")
    print(f"  {'-' * 60}")
    for w1, w2 in similar_pairs:
        if w1 in word_to_vec and w2 in word_to_vec:
            sim = cosine_similarity(word_to_vec[w1], word_to_vec[w2])
            note = "✓ 语义相关" if sim > 0.5 else ""
            print(f"  {w1:<12} {w2:<12} {sim:<+14.6f} {note}")

    print(f"\n  ▸ 不相关词对：")
    print(f"  {'词1':<12} {'词2':<12} {'余弦相似度':<14}")
    print(f"  {'-' * 52}")
    for w1, w2 in unrelated_pairs:
        if w1 in word_to_vec and w2 in word_to_vec:
            sim = cosine_similarity(word_to_vec[w1], word_to_vec[w2])
            print(f"  {w1:<12} {w2:<12} {sim:<+14.6f}")

    print(f"\n  ▸ 反义词对：")
    print(f"  {'词1':<12} {'词2':<12} {'余弦相似度':<14}")
    print(f"  {'-' * 52}")
    for w1, w2 in antonym_pairs:
        if w1 in word_to_vec and w2 in word_to_vec:
            sim = cosine_similarity(word_to_vec[w1], word_to_vec[w2])
            print(f"  {w1:<12} {w2:<12} {sim:<+14.6f}")
    print()


def demo_nearest_neighbors(
    word_to_vec: dict[str, np.ndarray],
    word_to_vec_norm: dict[str, np.ndarray],
) -> None:
    """查找与目标词语义最相近的词。"""
    print()
    print("=" * 70)
    print("  [语义最近邻] Top-10 最相似词查询")
    print("=" * 70)
    print()

    queries = ["king", "apple", "computer", "deep", "science"]

    for word in queries:
        if word not in word_to_vec:
            continue
        vec = word_to_vec[word]
        neighbors = find_similar_words(
            vec, word_to_vec, topk=10, exclude={word}
        )
        print(f"  「{word}」的语义最近邻：")
        print(f"  {'#':<4} {'词':<16} {'相似度':<12}")
        print(f"  {'-' * 35}")
        for i, (nw, sim) in enumerate(neighbors, 1):
            print(f"  {i:<4} {nw:<16} {sim:<+12.6f}")
        print()


def demo_word_analogies(
    word_to_vec: dict[str, np.ndarray],
    word_to_vec_norm: dict[str, np.ndarray],
) -> None:
    """词类比推理演示。"""
    print()
    print("=" * 70)
    print("  [词类比推理] A : B = C : ?")
    print("=" * 70)
    print(f"  核心公式：vec(B) - vec(A) + vec(C) → 找最近邻")
    print(f"  这是词嵌入最迷人的性质之一——语义偏移编码了关系！")
    print()

    analogies = [
        ("king",   "man",    "woman",    "王 : 男人 = 女人 : ?"),
        ("paris",  "france", "italy",    "巴黎 : 法国 = 意大利 : ?"),
        ("big",    "bigger", "small",    "大 : 更大 = 小 : ?"),
        ("walk",   "walked", "jump",     "走路 : 走了 = 跳 : ?"),
        ("dog",    "dogs",   "cat",      "狗 : 狗们 = 猫 : ?"),
        ("london", "england","japan",    "伦敦 : 英格兰 = 日本 : ?"),
        ("father", "mother", "uncle",    "父亲 : 母亲 = 叔叔 : ?"),
        ("cold",   "colder", "warm",     "冷 : 更冷 = 暖 : ?"),
        ("apple",  "fruit",  "rose",     "苹果 : 水果 = 玫瑰 : ?"),
        ("three",  "three",  "one",      "三 : 三 = 一 : ?"),
    ]

    for a, b, c, desc in analogies:
        if a not in word_to_vec or b not in word_to_vec or c not in word_to_vec:
            continue
        results = word_analogy(a, b, c, word_to_vec, word_to_vec_norm, topk=5)
        if results:
            top_word, top_sim = results[0]
            print(f"  {desc} → 预测：{top_word}（相似度={top_sim:.4f}）")
            others = ", ".join(f"{w}({s:.3f})" for w, s in results[1:3])
            if others:
                print(f"    其他候选：{others}")
            print()

    # 展示类比推理的原理
    print(f"  ── 原理说明 ──")
    print(f"  计算 'paris - france + italy' 的向量：")
    if all(w in word_to_vec for w in ["paris", "france", "italy"]):
        v_paris = word_to_vec["paris"]
        v_france = word_to_vec["france"]
        v_italy = word_to_vec["italy"]
        v_result = v_paris - v_france + v_italy

        def format_vec(v):
            return ", ".join(f"{x:+.3f}" for x in v[:6]) + ", ..."

        print(f"    vec(paris)    = [{format_vec(v_paris)}]")
        print(f"    vec(france)   = [{format_vec(v_france)}]")
        print(f"    vec(italy)    = [{format_vec(v_italy)}]")
        print(f"    ─────────────────────────────────")
        print(f"    result vector = [{format_vec(v_result)}]")
        print(f"    (paris - france + italy)")
        print()

        # 找最近邻验证
        nn = find_similar_words(v_result, word_to_vec, topk=5, exclude={"paris", "france", "italy"})
        print(f"    result 的最近邻：")
        for i, (w, s) in enumerate(nn, 1):
            print(f"      {i}. {w:<12} (相似度={s:.4f})")
        print()


def demo_custom_analogy(
    word_to_vec: dict[str, np.ndarray],
    word_to_vec_norm: dict[str, np.ndarray],
) -> None:
    """用户可自定义的词类比查询。"""
    print()
    print("=" * 70)
    print("  [自定义类比探索] 你可以修改下面的词对来探索")
    print("=" * 70)
    print()

    # ── 编辑这里的词对来探索不同的语义关系 ───────────────
    custom_pairs = [
        ("foot", "feet", "tooth"),            # 复数形态 → 预测 teeth
        ("eat", "ate", "drink"),              # 过去式 → 预测 drank
        ("good", "better", "bad"),            # 比较级 → 预测 worse
        ("france", "wine", "germany"),        # 国家:特产 → 预测 beer
        ("japan", "tokyo", "china"),          # 国家:首都 → 预测 beijing
        ("king", "prince", "queen"),          # 王:王子 = 后:公主
    ]
    # ────────────────────────────────────────────────────

    for a, b, c in custom_pairs:
        if a not in word_to_vec or b not in word_to_vec or c not in word_to_vec:
            continue
        results = word_analogy(a, b, c, word_to_vec, word_to_vec_norm, topk=5)
        if results:
            top_word, top_sim = results[0]
            print(f"  {a:>8} : {b:<8} = {c:>8} : ?  →  {top_word:<10} (sim={top_sim:.4f})")
            others = ", ".join(f"{w}({s:.3f})" for w, s in results[1:3])
            if others:
                print(f"    └ 备选：{others}")
    print()


def demo_vector_arithmetic(
    word_to_vec: dict[str, np.ndarray],
) -> None:
    """展示词向量算术运算。"""
    print()
    print("=" * 70)
    print("  [向量运算] 词向量的加减法展示语义组合")
    print("=" * 70)
    print()

    # ── 向量加法：组合语义 ──
    print("  ▸ 向量加法（语义组合）：")
    composition_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("apple", "orange"),
        ("computer", "laptop"),
    ]

    print(f"  {'词1':<12} {'词2':<12} {'组合向量最近邻':<30}")
    print(f"  {'-' * 56}")
    for w1, w2 in composition_pairs:
        if w1 not in word_to_vec or w2 not in word_to_vec:
            continue
        combined = word_to_vec[w1] + word_to_vec[w2]
        nn = find_similar_words(combined, word_to_vec, topk=3, exclude={w1, w2})
        nn_str = ", ".join(f"{w}({s:.3f})" for w, s in nn)
        print(f"  {w1:<12} {w2:<12} {nn_str:<30}")

    # ── 向量差：语义关系 ──
    print(f"\n  ▸ 向量差（关系提取）：")
    diff_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("france", "paris"),
        ("italy", "rome"),
    ]

    print(f"  {'词1':<12} {'词2':<12} {'差向量最近邻':<30} {'可能关系'}")
    print(f"  {'-' * 70}")
    for w1, w2 in diff_pairs:
        if w1 not in word_to_vec or w2 not in word_to_vec:
            continue
        diff = word_to_vec[w1] - word_to_vec[w2]
        nn = find_similar_words(diff, word_to_vec, topk=3, exclude={w1, w2})
        nn_str = ", ".join(f"{w}({s:.3f})" for w, s in nn)
        print(f"  {w1:<12} {w2:<12} {nn_str:<30}")
    print()

    # ── 向量长度与词频/抽象度的关系 ──
    print(f"  ▸ 向量 L2 范数分布：")
    norms = [(w, np.linalg.norm(vec)) for w, vec in word_to_vec.items()]
    norms.sort(key=lambda x: x[1], reverse=True)

    print(f"  L2 范数最大的词（表示更具体/高频？）：")
    for w, n in norms[:8]:
        print(f"    {w:<16} norm={n:.4f}")
    print()
    print(f"  L2 范数最小的词（表示更抽象/低频？）：")
    for w, n in norms[-8:]:
        print(f"    {w:<16} norm={n:.4f}")
    print()


def demo_visualization(
    word_to_vec: dict[str, np.ndarray],
    vectors: np.ndarray,
    words: list[str],
    output_dir: str | None = None,
) -> str | None:
    """
    PCA 降维可视化语义空间。
    返回保存的图片路径，或 None 如果失败。
    """
    # ── 选择一批有代表性的词 ──
    categories = {
        "国家":    ["china", "japan", "korea", "india", "france", "germany",
                    "italy", "spain", "russia", "brazil", "australia", "canada"],
        "首都":    ["beijing", "tokyo", "seoul", "paris", "berlin", "rome",
                    "madrid", "moscow", "london", "washington"],
        "水果":    ["apple", "banana", "orange", "grape", "mango", "pear",
                    "cherry", "strawberry", "watermelon", "lemon"],
        "动物":    ["dog", "cat", "tiger", "lion", "elephant", "horse",
                    "cow", "sheep", "wolf", "bear"],
        "情感":    ["love", "hate", "happy", "sad", "anger", "fear",
                    "joy", "surprise", "trust", "disgust"],
    }

    # 只保留词表中存在的词
    valid: dict[str, list[str]] = {}
    for cat, cat_words in categories.items():
        valid_words = [w for w in cat_words if w in word_to_vec]
        if len(valid_words) >= 3:
            valid[cat] = valid_words

    if not valid:
        print("  ✗ 没有足够的词用于可视化")
        return None

    # 收集所有需要可视化的词及其向量
    viz_words: list[str] = []
    viz_indices: list[int] = []
    viz_categories: list[str] = []
    word_to_cat: dict[str, str] = {}

    for cat, cat_words in valid.items():
        for w in cat_words:
            if w in word_to_vec and w not in word_to_cat:
                vidx = words.index(w)
                viz_words.append(w)
                viz_indices.append(vidx)
                viz_categories.append(cat)
                word_to_cat[w] = cat

    if len(viz_words) < 5:
        print("  ✗ 可视化词太少")
        return None

    # PCA 降维
    selected_vectors = vectors[viz_indices]
    projected = pca_2d(selected_vectors)

    # 创建颜色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    cat_color = {}
    for i, cat in enumerate(valid.keys()):
        cat_color[cat] = colors[i]

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("white")

    # 设置中文字体
    chinese_font = None
    try:
        # Windows 常见中文字体
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "/System/Library/Fonts/PingFang.ttc", # macOS
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf",  # Linux
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                chinese_font = fm.FontProperties(fname=fp)
                break
    except Exception:
        pass

    # 逐类别绘制
    for cat in valid:
        cat_words_list = valid[cat]
        xs = []
        ys = []
        labels = []
        for w in cat_words_list:
            if w in word_to_cat and word_to_cat[w] == cat:
                vidx = viz_words.index(w)
                xs.append(projected[vidx, 0])
                ys.append(projected[vidx, 1])
                labels.append(w)

        ax.scatter(xs, ys, color=cat_color[cat], s=120, alpha=0.8, label=cat,
                 edgecolors="white", linewidth=0.5)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                fontproperties=chinese_font,
                alpha=0.9,
            )

    ax.set_xlabel("PC1 (主成分 1)", fontsize=11, fontproperties=chinese_font)
    ax.set_ylabel("PC2 (主成分 2)", fontsize=11, fontproperties=chinese_font)
    ax.set_title(
        "GloVe 词嵌入语义空间可视化（PCA 降维到 2D）",
        fontsize=14, fontweight="bold", fontproperties=chinese_font,
    )
    ax.legend(
        loc="best", fontsize=10, prop=chinese_font,
        framealpha=0.9, edgecolor="#cccccc",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#f8f8f8")

    # 添加说明
    fig.text(
        0.5, 0.01,
        "语义相近的词在嵌入空间中聚集成簇 | "
        "坐标轴含义：PCA 降维后保留方差最大的两个方向",
        ha="center", fontsize=10, fontproperties=chinese_font,
        color="#666666",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # 保存图片
    if output_dir is None:
        output_dir = str(CACHE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "glove_embedding_visualization.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ 可视化图片已保存：{img_path}")
    return img_path


def demo_coverage(word_to_vec: dict[str, np.ndarray]) -> None:
    """展示 GloVe 词表对常见英文词汇的覆盖情况。"""
    print()
    print("=" * 70)
    print("  [词表覆盖] 测试常见词汇在 GloVe 中的覆盖情况")
    print("=" * 70)
    print()

    test_sets = {
        "基础英语 (Top 50)": [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        ],
        "技术术语": [
            "algorithm", "neural", "network", "deep", "learning",
            "transformer", "attention", "embedding", "tokenizer", "gradient",
            "backpropagation", "convolution", "normalization", "optimization",
        ],
        "专有名词": [
            "google", "microsoft", "apple", "amazon", "facebook",
            "python", "java", "linux", "windows", "tensorflow",
            "pytorch", "nvidia", "intel", "ibm", "twitter",
        ],
        "中文拼音": [
            "beijing", "shanghai", "guangzhou", "shenzhen", "hangzhou",
            "tianjin", "chengdu", "nanjing", "wuhan", "xian",
        ],
    }

    for set_name, words in test_sets.items():
        found = sum(1 for w in words if w in word_to_vec)
        missing = [w for w in words if w not in word_to_vec]
        print(f"  ▸ {set_name}：")
        print(f"      覆盖：{found}/{len(words)}（{found/len(words)*100:.0f}%）")
        if missing:
            print(f"      缺失：{', '.join(missing)}")
        print()


def demo_embedding_gap(word_to_vec: dict[str, np.ndarray]) -> None:
    """
    展示静态词嵌入的局限性 —— 一词多义问题（polysemy）。
    GloVe 为每个词只分配一个向量，无法区分不同含义。
    """
    print()
    print("=" * 70)
    print("  [静态嵌入的局限] 一词多义（Polysemy）问题")
    print("=" * 70)
    print()
    print(f"  GloVe 为每个词只学习一个固定的向量表示。")
    print(f"  但很多词有多种含义，同一个向量需要同时编码所有含义。")
    print()

    polysemy_words = ["bank", "bat", "light", "fair", "bow", "seal", "spring"]

    print(f"  {'词':<10} {'词向量最近邻（混合了多种含义）':<60}")
    print(f"  {'-' * 72}")
    for word in polysemy_words:
        if word not in word_to_vec:
            continue
        nn = find_similar_words(word_to_vec[word], word_to_vec, topk=8, exclude={word})
        nn_str = ", ".join(f"{w}" for w, _ in nn)
        print(f"  {word:<10} {nn_str:<60}")

    print()
    print(f"  例如「bank」的近邻同时包含「金融机构」（money, deposit, account）和")
    print(f"  「河岸」（river, shore, coast）两种含义——因为同一个向量必须兼顾两者。")
    print(f"  这就是 2018 年后上下文相关嵌入（BERT、ELMo）替代静态嵌入的原因之一。")
    print()


# ─────────────────────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────────────────────

def main():
    # ── 步骤 1：加载 GloVe 词向量 ──
    print()
    print("=" * 70)
    print("  GloVe 词嵌入模型演示 — glove.6B.50d")
    print("  静态词向量的语义空间探索")
    print("=" * 70)
    print()

    print("[步骤 1/2] 加载 GloVe 预训练词向量...")
    glove_file = ensure_glove_file()
    word_to_vec, vectors, words = load_glove(glove_file)

    # 预计算归一化向量，加速后续相似度搜索
    vectors_norm = normalize_vectors(vectors)
    word_to_vec_norm = {}
    for i, w in enumerate(words):
        word_to_vec_norm[w] = vectors_norm[i]

    print("\n[步骤 2/2] 开始演示...")
    print()

    # ── 演示内容 ──
    demo_basic_info(word_to_vec)           # 基本信息
    demo_word_embeddings(word_to_vec)      # 词向量详解
    demo_similarity(word_to_vec)           # 语义相似度
    demo_nearest_neighbors(word_to_vec, word_to_vec_norm)   # 最近邻
    demo_word_analogies(word_to_vec, word_to_vec_norm)      # 词类比
    demo_custom_analogy(word_to_vec, word_to_vec_norm)      # 自定义类比
    demo_vector_arithmetic(word_to_vec)    # 向量运算
    demo_coverage(word_to_vec)             # 词表覆盖
    demo_embedding_gap(word_to_vec)        # 局限性

    # ── 可视化（保存图片） ──
    print("=" * 70)
    print("  [可视化] PCA 降维到 2D 展示语义空间")
    print("=" * 70)
    print()
    img_path = demo_visualization(word_to_vec, vectors, words)
    if img_path:
        print(f"\n  💡 可视化图片保存在：{img_path}")
        print(f"     可打开查看词向量在语义空间中的分布。")
    print()

    # ── 总结 ──
    print("=" * 70)
    print("  [总结] GloVe 静态词嵌入")
    print("=" * 70)
    print(f"  1. 词嵌入是将词映射到稠密向量空间的技术。")
    print(f"  2. GloVe 利用全局共现统计学习词向量，比 Word2Vec 更充分地")
    print(f"     利用了语料库的全局信息。")
    print(f"  3. 词向量编码了丰富的语义关系：")
    print(f"     - 语义相似度 → 向量夹角余弦")
    print(f"     - 语义类比 → 向量偏移（king - man + woman ≈ queen）")
    print(f"     - 语义组合 → 向量加法")
    print(f"  4. 局限性：")
    print(f"     - 每个词只有一种表示（无法处理一词多义）")
    print(f"     - 无法处理 OOV（词表外词）")
    print(f"     - 向量是静态的，不随上下文变化")
    print(f"  5. 后续发展：ELMo（上下文感知）→ BERT（双向上下文）→ ")
    print(f"     LLM 内部隐层表示作为动态嵌入")
    print()


if __name__ == "__main__":
    main()
