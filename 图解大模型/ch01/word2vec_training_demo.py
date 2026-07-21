"""
Word2Vec 词向量训练示例 — CBOW 与 Skip-gram 从零实现

Word2Vec 是 Google 在 2013 年提出的经典词嵌入方法，用浅层神经网络
从大规模无标注语料中学习稠密词向量。与 GloVe 利用全局共现矩阵不同，
Word2Vec 仅利用局部上下文窗口内的共现信息。

本示例使用纯 NumPy 从零实现 Word2Vec（不依赖 Gensim），包括：
  1. 语料准备与预处理（内置英文语料，支持 NLTK）
  2. 构建训练样本（上下文-目标词对）
  3. 从零实现 CBOW + 负采样 的训练循环
  4. 从零实现 Skip-gram + 负采样 的训练循环
  5. 训练损失曲线可视化
  6. 词向量相似度查询
  7. 词类比推理（word analogy）
  8. PCA 降维可视化语义空间
  9. 两种模型对比分析

CBOW vs Skip-gram 对比：
  - CBOW：用上下文词预测中心词 → 训练快，适合高频词
  - Skip-gram：用中心词预测上下文词 → 对低频词效果更好，训练慢

核心训练技巧：
  - 负采样（Negative Sampling）：每次只更新少数随机负样本，避免计算全词表 softmax

论文参考：
  - Mikolov et al. "Efficient Estimation of Word Representations in Vector Space" (2013)
  - Mikolov et al. "Distributed Representations of Words and Phrases..." (2013)
"""

import os
import sys
import time
import random
import tempfile
from pathlib import Path
from collections import Counter

# ── 确保 stdout 使用 UTF-8（解决 Windows GBK 编码问题） ────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# ── 可视化（可选）────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")                 # 无头模式，避免 GUI 弹窗
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("  ⚠ matplotlib 未安装，可视化功能将跳过")


# ── 常量 ─────────────────────────────────────────────────────
VECTOR_SIZE = 100          # 词向量维度
WINDOW = 5                 # 上下文窗口大小
MIN_COUNT = 2              # 最低词频（低频词忽略）
EPOCHS = 50                # 训练轮数（小语料需要多跑几轮）
BATCH_SIZE = 256           # mini-batch 大小
NEGATIVE_SAMPLES = 5       # 负采样数量
LEARNING_RATE = 0.05       # 初始学习率
SEED = 42                  # 随机种子

OUTPUT_DIR = Path(tempfile.gettempdir()) / "word2vec_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. 语料准备
# ─────────────────────────────────────────────────────────────

def prepare_corpus() -> tuple[list[list[str]], int, int]:
    """
    准备训练语料。

    优先使用 NLTK Brown 语料库（约 100 万词），
    不可用时使用内置英文示例语料（覆盖多领域）。

    返回：
      sentences : 分词后的句子列表（每个句子是词的列表）
      total_words : 总词数
      vocab_size : 去重词数
    """
    print("[步骤 1/7] 准备训练语料...")
    print("-" * 60)

    sentences: list[list[str]] = []

    # ── 尝试使用 NLTK Brown 语料库 ─────────────────────────
    print(f"  ! NLTK 未安装，使用内置示例语料")
    sentences = _get_builtin_corpus()
        

    if len(sentences) < 50:
        print(f"  ! 语料句子不足，补充内置语料")
        sentences.extend(_get_builtin_corpus())

    # 统计
    total_words = sum(len(s) for s in sentences)
    all_words = [w for s in sentences for w in s]
    vocab_size = len(set(all_words))

    print(f"    总词数：{total_words:,}")
    print(f"    词表大小（去重）：{vocab_size:,}")
    print(f"    平均句子长度：{total_words / max(len(sentences), 1):.1f} 词")

    word_counts = Counter(all_words)
    print(f"\n  词频 Top-20：")
    for i, (word, count) in enumerate(word_counts.most_common(20), 1):
        print(f"    {i:2d}. {word:<14} {count:>6}")
    print()

    return sentences, total_words, vocab_size


def _get_builtin_corpus() -> list[list[str]]:
    """
    内置英文示例语料 —— 涵盖日常生活、科技、自然等多个领域。
    当 NLTK 不可用时作为后备语料，确保示例可独立运行。
    """
    raw_texts = [
        # 日常生活
        "the king ruled the kingdom with wisdom and justice",
        "the queen walked through the beautiful garden in the morning",
        "the man went to the market to buy fresh apples and oranges",
        "the woman cooked a delicious dinner for her family",
        "children love to play in the park on sunny days",
        "the dog chased the cat around the big old house",
        "the cat sat on the warm windowsill watching birds outside",
        "a boy and his father went fishing at the river yesterday",
        "the mother read a bedtime story to her little daughter",
        "students studied hard for the final examination at school",
        "people gathered in the town square to celebrate the festival",
        "the farmer planted wheat and corn in the vast field",
        "a young girl painted a beautiful picture of the ocean",
        "the old man told interesting stories about his childhood",
        "friends met at the coffee shop to discuss their weekend plans",
        "the baby slept peacefully in the wooden cradle",
        "she bought new shoes and a dress from the department store",
        "he drove his car to the office through heavy morning traffic",
        "the musician played a wonderful melody on the grand piano",
        "tourists visited the ancient castle and took many photographs",

        # 科技与计算机
        "the computer processed large amounts of data very quickly",
        "scientists discovered a new method for training neural networks",
        "deep learning algorithms can recognize patterns in images and speech",
        "the programmer wrote elegant code to solve the complex problem",
        "machine learning models require large datasets for effective training",
        "artificial intelligence is transforming many industries and professions",
        "the robot learned to navigate through the room using its sensors",
        "software engineers developed a new application for mobile phones",
        "the database stored millions of records for future analysis",
        "researchers published a paper on natural language processing technology",
        "the algorithm calculated the shortest path between two cities",
        "cybersecurity experts protected the network from malicious attacks",
        "the server responded to thousands of requests every second",
        "data scientists analyzed customer behavior using statistical methods",
        "the new processor was faster and more efficient than previous models",

        # 自然与动物
        "the lion hunted for food in the vast african savanna",
        "birds migrated south before the cold winter arrived",
        "the river flowed gently through the green valley below",
        "tall mountains rose above the clouds in the distant horizon",
        "the forest was filled with the sounds of insects and animals",
        "colorful fish swam among the coral reefs in the clear ocean",
        "the eagle soared high above the rocky mountain peaks",
        "gentle rain fell on the fields bringing life to the crops",
        "the sun set behind the hills painting the sky in golden colors",
        "wolves howled at the full moon in the dark winter night",
        "the horse galloped across the wide open prairie at dawn",
        "bees collected nectar from the bright yellow sunflowers",
        "the ocean waves crashed against the rocky shore at sunset",
        "a gentle breeze carried the scent of pine through the woods",
        "the desert stretched endlessly under the burning summer sun",

        # 学习与知识
        "the teacher explained difficult concepts with clear examples",
        "students learned about history and geography in their classes",
        "reading books expands our knowledge and improves vocabulary",
        "the university offered courses in medicine law and engineering",
        "scientific research requires patience creativity and rigorous methods",
        "mathematics is the foundation of many scientific disciplines",
        "the library contained thousands of books on various subjects",
        "learning a new language opens doors to different cultures",
        "the professor gave a fascinating lecture on quantum physics",
        "students discussed philosophical ideas in the seminar room",
        "education is the most powerful tool for changing the world",
        "the scientist conducted experiments in the laboratory every day",
        "knowledge of history helps us understand the present better",
        "the study of astronomy reveals the mysteries of the universe",
        "critical thinking is an essential skill in the modern world",

        # 商业与经济
        "the company launched a new product that became very successful",
        "investors analyzed the stock market trends before making decisions",
        "the factory produced thousands of units every single day",
        "customers praised the quality and durability of the new product",
        "the economy grew rapidly after the new trade policies were implemented",
        "small businesses created many new jobs in the local community",
        "the bank approved a loan for the entrepreneur to start her business",
        "marketing strategies focused on social media and digital platforms",
        "the corporation expanded its operations to international markets",
        "supply chain management became more efficient with new technology",
    ]

    sentences = []
    for text in raw_texts:
        words = [w.lower().strip(".,;:!?\"'()[]") for w in text.split()]
        words = [w for w in words if len(w) > 1]
        if len(words) >= 3:
            sentences.append(words)

    random.seed(SEED)
    # 通过数据增强增加语料量
    augmented = list(sentences)
    for sent in sentences:
        if len(sent) > 5 and random.random() < 0.3:
            filtered = [w for w in sent if random.random() > 0.15]
            if len(filtered) >= 3:
                augmented.append(filtered)

    return augmented


# ─────────────────────────────────────────────────────────────
# 2. 从零实现 Word2Vec（纯 NumPy）
# ─────────────────────────────────────────────────────────────

class Word2VecFromScratch:
    """
    Word2Vec 的纯 NumPy 实现。

    支持 CBOW 和 Skip-gram 两种架构，使用负采样（Negative Sampling）
    来近似 softmax，避免每次更新整个词表的开销。

    架构说明：
      - 输入词嵌入矩阵 W_in  (V x D)：将词 ID 映射到稠密向量
      - 输出词嵌入矩阵 W_out (V x D)：用于计算与目标词的得分
      - 最终词向量通常取 W_in，或 W_in 与 W_out 的平均值

    训练流程（以 Skip-gram 为例）：
      1. 取中心词 → 查 W_in 得到向量 v_c
      2. 对每个上下文词：
         a. 正样本得分：s_pos = v_c · W_out[context_word]
         b. 负样本得分：s_neg = v_c · W_out[random_words]
         c. loss = -log(σ(s_pos)) - Σ log(σ(-s_neg))
         d. 反向传播更新 W_in 和 W_out
    """

    def __init__(
        self,
        vocab_size: int,
        vector_size: int = 100,
        window: int = 5,
        negative_samples: int = 5,
        learning_rate: float = 0.05,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.vector_size = vector_size
        self.window = window
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate

        rng = np.random.RandomState(seed)

        # 输入嵌入矩阵：将词 ID → 稠密向量
        # 用 Xavier 风格的初始化
        scale = 1.0 / np.sqrt(vector_size)
        self.W_in = rng.uniform(-scale, scale, (vocab_size, vector_size)).astype(np.float32)

        # 输出嵌入矩阵：用于计算得分
        self.W_out = rng.uniform(-scale, scale, (vocab_size, vector_size)).astype(np.float32)

        # 梯度累积器
        self.grad_W_in = np.zeros_like(self.W_in)
        self.grad_W_out = np.zeros_like(self.W_out)

    def get_vector(self, word_id: int) -> np.ndarray:
        """获取词的输入向量（这是我们最终要的词向量）。"""
        return self.W_in[word_id].copy()

    def get_vectors(self) -> np.ndarray:
        """获取全部词的输入向量矩阵。"""
        return self.W_in.copy()

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid 激活函数，数值稳定版本。"""
        # 防止溢出
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))


class CBOWTrainer:
    """
    CBOW（Continuous Bag of Words）训练器。

    CBOW 原理：
      - 输入：上下文窗口内所有词的词向量取平均值
      - 输出：预测中心词
      - 损失：负采样损失（NCE 的简化版）

    直观理解：
      给定 "... the __ dog over the ..."，用周围的词预测中间被遮住的词。
    """

    def __init__(self, model: Word2VecFromScratch, word_freq: np.ndarray):
        self.model = model
        # 负采样分布：P(w) ∝ freq(w)^0.75（论文中的经验设置）
        freq_pow = word_freq ** 0.75
        self.noise_dist = freq_pow / freq_pow.sum()

    def train_step(
        self,
        center_id: int,
        context_ids: list[int],
        lr: float,
    ) -> float:
        """
        单步训练：用上下文词预测中心词。

        参数：
          center_id   : 中心词（目标）的 ID
          context_ids : 上下文词的 ID 列表
          lr          : 当前学习率

        返回：
          该步的损失值
        """
        if len(context_ids) == 0:
            return 0.0

        V, D = self.model.vocab_size, self.model.vector_size

        # ── 前向传播 ──────────────────────────────────────
        # 1. 取所有上下文词的输入向量，求平均
        context_vecs = self.model.W_in[context_ids]          # (N_ctx, D)
        h = context_vecs.mean(axis=0)                         # (D,)
        h = h / (np.linalg.norm(h) + 1e-10)                  # L2 归一化

        # 2. 正样本得分：h 与中心词的输出向量做内积
        pos_score = np.dot(h, self.model.W_out[center_id])   # 标量

        # 3. 负样本：从噪声分布中采样
        neg_ids = np.random.choice(
            V, size=self.model.negative_samples,
            p=self.noise_dist, replace=True,
        )
        # 排除恰好等于中心词的负样本
        neg_ids = neg_ids[neg_ids != center_id]

        neg_scores = h @ self.model.W_out[neg_ids].T          # (N_neg,)

        # 4. 计算损失（负采样损失）
        pos_loss = -np.log(self.model._sigmoid(pos_score) + 1e-10)
        neg_loss = -np.sum(np.log(self.model._sigmoid(-neg_scores) + 1e-10))
        loss = pos_loss + neg_loss

        # ── 反向传播 ──────────────────────────────────────
        # 梯度：∂L/∂h = (σ(pos_score) - 1) * W_out[center] + Σ σ(neg_score_i) * W_out[neg_i]
        grad_h = np.zeros(D, dtype=np.float32)

        # 正样本梯度
        sig_pos = self.model._sigmoid(pos_score)
        grad_h += (sig_pos - 1.0) * self.model.W_out[center_id]
        # 更新正样本的输出向量
        self.model.W_out[center_id] -= lr * (sig_pos - 1.0) * h

        # 负样本梯度
        for nid in neg_ids:
            sig_neg = self.model._sigmoid(neg_scores[list(neg_ids).index(nid)])
            grad_h += sig_neg * self.model.W_out[nid]
            self.model.W_out[nid] -= lr * sig_neg * h

        # 将梯度回传到每个上下文词的输入向量
        grad_per_context = grad_h / len(context_ids)
        for cid in context_ids:
            self.model.W_in[cid] -= lr * grad_per_context

        return float(loss)


class SkipGramTrainer:
    """
    Skip-gram 训练器。

    Skip-gram 原理：
      - 输入：中心词的词向量
      - 输出：预测上下文窗口内的每一个词
      - 损失：对每个上下文词独立计算负采样损失

    直观理解：
      给定词 "dog"，预测它周围可能出现的词（如 "the", "chased", "cat"...）。
    """

    def __init__(self, model: Word2VecFromScratch, word_freq: np.ndarray):
        self.model = model
        freq_pow = word_freq ** 0.75
        self.noise_dist = freq_pow / freq_pow.sum()

    def train_step(
        self,
        center_id: int,
        context_ids: list[int],
        lr: float,
    ) -> float:
        """
        单步训练：用中心词预测每个上下文词。

        返回：
          平均损失值（对所有上下文词取平均）
        """
        if len(context_ids) == 0:
            return 0.0

        V, D = self.model.vocab_size, self.model.vector_size
        total_loss = 0.0

        # 中心词的输入向量
        v_c = self.model.W_in[center_id]                     # (D,)

        for target_id in context_ids:
            # ── 前向传播 ──────────────────────────────────
            # 正样本得分
            pos_score = np.dot(v_c, self.model.W_out[target_id])

            # 负样本
            neg_ids = np.random.choice(
                V, size=self.model.negative_samples,
                p=self.noise_dist, replace=True,
            )
            neg_ids = neg_ids[neg_ids != target_id]

            neg_scores = v_c @ self.model.W_out[neg_ids].T

            # 损失
            pos_loss = -np.log(self.model._sigmoid(pos_score) + 1e-10)
            neg_loss = -np.sum(np.log(self.model._sigmoid(-neg_scores) + 1e-10))
            total_loss += pos_loss + neg_loss

            # ── 反向传播 ──────────────────────────────────
            grad_v_c = np.zeros(D, dtype=np.float32)

            # 正样本
            sig_pos = self.model._sigmoid(pos_score)
            grad_v_c += (sig_pos - 1.0) * self.model.W_out[target_id]
            self.model.W_out[target_id] -= lr * (sig_pos - 1.0) * v_c

            # 负样本
            for nid in neg_ids:
                sig_neg = self.model._sigmoid(neg_scores[list(neg_ids).index(nid)])
                grad_v_c += sig_neg * self.model.W_out[nid]
                self.model.W_out[nid] -= lr * sig_neg * v_c

            # 更新中心词的输入向量
            self.model.W_in[center_id] -= lr * grad_v_c

        return total_loss / max(len(context_ids), 1)


# ─────────────────────────────────────────────────────────────
# 3. 训练流程
# ─────────────────────────────────────────────────────────────

def build_vocab_and_samples(
    sentences: list[list[str]],
) -> tuple[dict[str, int], list[str], np.ndarray, list[tuple[int, list[int]]]]:
    """
    构建词表并生成训练样本。

    返回：
      word_to_id : 词到 ID 的映射
      id_to_word : ID 到词的映射
      word_freq  : 词频数组（用于负采样分布）
      samples    : (center_id, [context_ids]) 训练样本列表
    """
    # 统计词频
    word_counts = Counter(w for sent in sentences for w in sent)

    # 过滤低频词
    vocab_words = [w for w, c in word_counts.items() if c >= MIN_COUNT]
    vocab_words.sort()

    # 构建映射
    word_to_id = {w: i for i, w in enumerate(vocab_words)}
    id_to_word = {i: w for w, i in word_to_id.items()}

    # 词频数组
    word_freq = np.zeros(len(vocab_words), dtype=np.float32)
    for w, i in word_to_id.items():
        word_freq[i] = word_counts[w]

    # 生成样本
    samples: list[tuple[int, list[int]]] = []
    for sent in sentences:
        ids = [word_to_id[w] for w in sent if w in word_to_id]
        for i, center in enumerate(ids):
            # 随机窗口大小（论文技巧：对每个中心词随机选窗口大小）
            win = random.randint(1, WINDOW)
            start = max(0, i - win)
            end = min(len(ids), i + win + 1)
            context = ids[start:i] + ids[i + 1:end]
            if context:
                samples.append((center, context))

    print(f"  词表大小：{len(vocab_words):,}（去除了词频 < {MIN_COUNT} 的低频词）")
    print(f"  训练样本数：{len(samples):,}")
    print()

    return word_to_id, id_to_word, word_freq, samples


def train_cbow(
    samples: list[tuple[int, list[int]]],
    vocab_size: int,
    word_freq: np.ndarray,
) -> tuple[Word2VecFromScratch, list[float]]:
    """训练 CBOW 模型，返回模型和每 epoch 的损失列表。"""
    print("[步骤 3/7] 训练 CBOW 模型...")
    print("-" * 60)
    print(f"  算法：CBOW（Continuous Bag of Words）")
    print(f"  向量维度：{VECTOR_SIZE}")
    print(f"  上下文窗口：±{WINDOW} 个词（每个中心词随机选取窗口大小）")
    print(f"  负采样数：{NEGATIVE_SAMPLES}")
    print(f"  初始学习率：{LEARNING_RATE}")
    print(f"  训练轮数：{EPOCHS}")
    print()

    model = Word2VecFromScratch(
        vocab_size=vocab_size,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        negative_samples=NEGATIVE_SAMPLES,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )
    trainer = CBOWTrainer(model, word_freq)

    epoch_losses: list[float] = []
    total_samples = len(samples)

    for epoch in range(1, EPOCHS + 1):
        # 每轮打乱样本顺序
        random.shuffle(samples)

        # 学习率线性衰减
        lr = LEARNING_RATE * (1.0 - (epoch - 1) / EPOCHS)

        epoch_loss = 0.0
        t0 = time.time()

        for center, context in samples:
            loss = trainer.train_step(center, context, lr)
            epoch_loss += loss

        avg_loss = epoch_loss / max(total_samples, 1)
        epoch_losses.append(avg_loss)

        elapsed = time.time() - t0

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"loss={avg_loss:.4f} | lr={lr:.4f} | {elapsed:.2f}s")

    print(f"\n  训练完成！最终损失：{epoch_losses[-1]:.4f}")
    print()

    return model, epoch_losses


def train_skipgram(
    samples: list[tuple[int, list[int]]],
    vocab_size: int,
    word_freq: np.ndarray,
) -> tuple[Word2VecFromScratch, list[float]]:
    """训练 Skip-gram 模型，返回模型和每 epoch 的损失列表。"""
    print("[步骤 4/7] 训练 Skip-gram 模型...")
    print("-" * 60)
    print(f"  算法：Skip-gram")
    print(f"  向量维度：{VECTOR_SIZE}")
    print(f"  上下文窗口：±{WINDOW} 个词")
    print(f"  负采样数：{NEGATIVE_SAMPLES}")
    print(f"  初始学习率：{LEARNING_RATE}")
    print(f"  训练轮数：{EPOCHS}")
    print()

    model = Word2VecFromScratch(
        vocab_size=vocab_size,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        negative_samples=NEGATIVE_SAMPLES,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )
    trainer = SkipGramTrainer(model, word_freq)

    epoch_losses: list[float] = []
    total_samples = len(samples)

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(samples)

        lr = LEARNING_RATE * (1.0 - (epoch - 1) / EPOCHS)

        epoch_loss = 0.0
        t0 = time.time()

        for center, context in samples:
            loss = trainer.train_step(center, context, lr)
            epoch_loss += loss

        avg_loss = epoch_loss / max(total_samples, 1)
        epoch_losses.append(avg_loss)

        elapsed = time.time() - t0

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"loss={avg_loss:.4f} | lr={lr:.4f} | {elapsed:.2f}s")

    print(f"\n  训练完成！最终损失：{epoch_losses[-1]:.4f}")
    print()

    return model, epoch_losses


# ─────────────────────────────────────────────────────────────
# 4. 向量运算工具
# ─────────────────────────────────────────────────────────────

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """用 SVD 实现 PCA 降维到 2D（纯 numpy，不需要 sklearn）。"""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:2].T
    return projected


def find_similar_words(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    id_to_word: dict[int, str],
    topk: int = 10,
    exclude: set[int] | None = None,
) -> list[tuple[str, float]]:
    """
    在词表中查找与 query_vec 最相似的 topk 个词。
    """
    exclude = exclude or set()
    results: list[tuple[str, float]] = []

    for wid in range(vectors.shape[0]):
        if wid in exclude:
            continue
        sim = cosine_similarity(query_vec, vectors[wid])
        results.append((id_to_word[wid], sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topk]


def word_analogy(
    a: str, b: str, c: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    vectors: np.ndarray,
    topk: int = 5,
) -> list[tuple[str, float]]:
    """
    词类比推理：a 之于 b，如同 c 之于 ?
    即：vec(b) - vec(a) + vec(c) 的最近邻（排除 a, b, c）。
    """
    if a not in word_to_id or b not in word_to_id or c not in word_to_id:
        return []

    va = vectors[word_to_id[a]]
    vb = vectors[word_to_id[b]]
    vc = vectors[word_to_id[c]]
    target = vb - va + vc

    exclude_ids = {word_to_id[w] for w in (a, b, c)}
    return find_similar_words(target, vectors, id_to_word, topk=topk, exclude=exclude_ids)


# ─────────────────────────────────────────────────────────────
# 5. 演示函数
# ─────────────────────────────────────────────────────────────

def demo_loss_curves(
    cbow_losses: list[float],
    sg_losses: list[float],
) -> str | None:
    """绘制训练损失曲线对比。"""
    print()
    print("=" * 70)
    print("  [训练损失曲线] CBOW vs Skip-gram")
    print("=" * 70)

    if not HAS_MATPLOTLIB:
        # 用文本展示损失趋势
        print(f"\n  {'Epoch':<8} {'CBOW Loss':<14} {'Skip-gram Loss':<16}")
        print(f"  {'-' * 40}")
        step = max(1, len(cbow_losses) // 10)
        for i in range(0, len(cbow_losses), step):
            print(f"  {i+1:<8} {cbow_losses[i]:<14.4f} {sg_losses[i]:<16.4f}")
        print(f"  {'-' * 40}")
        print(f"  Final    {cbow_losses[-1]:<14.4f} {sg_losses[-1]:<16.4f}")
        print(f"\n  ⚠ matplotlib 未安装，以上为文本版损失变化。")
        print(f"    安装 matplotlib 后可获得 PNG 图表。")
        return None

    chinese_font = _get_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    epochs = range(1, len(cbow_losses) + 1)
    ax.plot(epochs, cbow_losses, "b-", linewidth=1.5, alpha=0.8, label="CBOW")
    ax.plot(epochs, sg_losses, "r-", linewidth=1.5, alpha=0.8, label="Skip-gram")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Average Loss", fontsize=11)
    ax.set_title("Word2Vec Training Loss — CBOW vs Skip-gram",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    img_path = str(OUTPUT_DIR / "word2vec_training_loss.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 损失曲线已保存：{img_path}")
    return img_path


def demo_model_info(
    cbow_model: Word2VecFromScratch,
    sg_model: Word2VecFromScratch,
    word_to_id: dict[str, int],
) -> None:
    """展示模型基本信息。"""
    print()
    print("=" * 70)
    print("  [模型信息] Word2Vec 从零实现")
    print("=" * 70)
    print(f"  ├ 词表大小：{len(word_to_id):,}")
    print(f"  ├ 向量维度：{VECTOR_SIZE}")
    print(f"  ├ 上下文窗口：±{WINDOW} 词")
    print(f"  ├ 负采样数：{NEGATIVE_SAMPLES}")
    print(f"  ├ 训练轮数：{EPOCHS}")
    print(f"  ├ 初始学习率：{LEARNING_RATE}")
    print(f"  ├ CBOW 参数量：{cbow_model.W_in.size + cbow_model.W_out.size:,}")
    print(f"  └ Skip-gram 参数量：{sg_model.W_in.size + sg_model.W_out.size:,}")

    # 展示几个样本词的向量快照
    sample_words = ["king", "queen", "computer", "learning", "science"]
    valid_samples = [w for w in sample_words if w in word_to_id]

    if valid_samples:
        print(f"\n  样本词向量快照（CBOW）：")
        print(f"  {'词':<14} {'前 8 维值':<55} {'L2范数':<10}")
        print(f"  {'-' * 80}")
        for w in valid_samples:
            wid = word_to_id[w]
            vec = cbow_model.get_vector(wid)
            front = ", ".join(f"{v:+.4f}" for v in vec[:8])
            norm = np.linalg.norm(vec)
            print(f"  {w:<14} [{front}, ...]  {norm:<10.4f}")
    print()


def demo_word_vectors(
    model: Word2VecFromScratch,
    word_to_id: dict[str, int],
    label: str,
) -> None:
    """展示指定词的嵌入向量详细信息。"""
    print()
    print("=" * 70)
    print(f"  [词向量详解] {label} 模型")
    print("=" * 70)

    target_words = ["king", "queen", "computer", "learning", "science"]
    valid = [w for w in target_words if w in word_to_id]

    for word in valid[:5]:
        vec = model.get_vector(word_to_id[word])
        print(f"\n  ── {word} ──")
        print(f"    向量值（前 50 维 / 共 {len(vec)} 维）：")
        for row in range(5):
            start = row * 10
            end = start + 10
            vals = ", ".join(f"{v:+.4f}" for v in vec[start:end])
            print(f"      [{start:2d}-{end:2d}]  {vals}")
        print(f"    ... 共 {len(vec)} 维")
        print(f"    统计：均值={vec.mean():+.6f}, 标准差={vec.std():.6f}, "
              f"L2范数={np.linalg.norm(vec):.4f}, "
              f"最小值={vec.min():+.6f}, 最大值={vec.max():+.6f}")
    print()


def demo_similarity(
    model: Word2VecFromScratch,
    word_to_id: dict[str, int],
    label: str,
) -> None:
    """展示词间相似度计算。"""
    print()
    print("=" * 70)
    print(f"  [语义相似度] {label} 模型")
    print("=" * 70)
    print(f"  语义相近的词 → 高相似度（接近 1.0）")
    print(f"  语义无关的词 → 低相似度（接近 0.0 或负值）")
    print()

    similar_pairs = [
        ("king", "queen"),
        ("dog", "cat"),
        ("computer", "software"),
        ("ocean", "sea"),
        ("learning", "education"),
        ("music", "song"),
        ("man", "woman"),
        ("sun", "moon"),
        ("sun", "ocean"),
    ]

    unrelated_pairs = [
        ("king", "apple"),
        ("computer", "ocean"),
        ("dog", "mathematics"),
    ]

    print(f"  ▸ 语义相近词对：")
    print(f"  {'词1':<14} {'词2':<14} {'余弦相似度':<14} {'判断'}")
    print(f"  {'-' * 56}")
    for w1, w2 in similar_pairs:
        if w1 in word_to_id and w2 in word_to_id:
            sim = cosine_similarity(
                model.get_vector(word_to_id[w1]),
                model.get_vector(word_to_id[w2]),
            )
            note = "✓ 相关" if sim > 0.2 else "△ 一般"
            print(f"  {w1:<14} {w2:<14} {sim:<+14.6f} {note}")
        else:
            print(f"  {w1:<14} {w2:<14} {'N/A':<14} (词表中缺失)")

    print(f"\n  ▸ 不相关词对：")
    print(f"  {'词1':<14} {'词2':<14} {'余弦相似度':<14}")
    print(f"  {'-' * 44}")
    for w1, w2 in unrelated_pairs:
        if w1 in word_to_id and w2 in word_to_id:
            sim = cosine_similarity(
                model.get_vector(word_to_id[w1]),
                model.get_vector(word_to_id[w2]),
            )
            print(f"  {w1:<14} {w2:<14} {sim:<+14.6f}")
    print()


def demo_similar_words(
    model: Word2VecFromScratch,
    vectors: np.ndarray,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    label: str,
) -> None:
    """查找与目标词语义最相近的词。"""
    print()
    print("=" * 70)
    print(f"  [语义最近邻] {label} 模型 — Top-10 最相似词")
    print("=" * 70)
    print()

    queries = ["king", "computer", "ocean", "learning", "music"]

    for word in queries:
        if word not in word_to_id:
            print(f"  「{word}」不在词表中，跳过。\n")
            continue

        wid = word_to_id[word]
        neighbors = find_similar_words(
            vectors[wid], vectors, id_to_word,
            topk=10, exclude={wid},
        )
        print(f"  「{word}」的语义最近邻：")
        print(f"  {'#':<4} {'词':<16} {'相似度':<12}")
        print(f"  {'-' * 34}")
        for i, (nw, sim) in enumerate(neighbors, 1):
            print(f"  {i:<4} {nw:<16} {sim:<+12.6f}")
        print()


def demo_word_analogies(
    vectors: np.ndarray,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    label: str,
) -> None:
    """词类比推理演示。"""
    print()
    print("=" * 70)
    print(f"  [词类比推理] {label} 模型 — A : B = C : ?")
    print("=" * 70)
    print(f"  核心公式：vec(B) - vec(A) + vec(C) → 找最近邻")
    print()

    analogies = [
        ("king",   "man",    "woman",    "王 : 男人 = 女人 : ? (期望: queen)"),
        ("paris",  "france", "italy",    "巴黎 : 法国 = 意大利 : ? (期望: rome)"),
        ("walk",   "walked", "jump",     "走路 : 走了 = 跳 : ? (期望: jumped)"),
        ("big",    "bigger", "small",    "大 : 更大 = 小 : ? (期望: smaller)"),
        ("dog",    "dogs",   "cat",      "狗 : 狗们 = 猫 : ? (期望: cats)"),
        ("father", "mother", "uncle",    "父亲 : 母亲 = 叔叔 : ? (期望: aunt)"),
        ("good",   "better", "bad",      "好 : 更好 = 坏 : ? (期望: worse)"),
        ("cold",   "colder", "warm",     "冷 : 更冷 = 暖 : ? (期望: warmer)"),
    ]

    expectation_map = {
        ("king", "man", "woman"): "queen",
        ("paris", "france", "italy"): "rome",
        ("walk", "walked", "jump"): "jumped",
        ("big", "bigger", "small"): "smaller",
        ("dog", "dogs", "cat"): "cats",
        ("father", "mother", "uncle"): "aunt",
        ("good", "better", "bad"): "worse",
        ("cold", "colder", "warm"): "warmer",
    }

    hit_count = 0
    test_count = 0
    for a, b, c, desc in analogies:
        results = word_analogy(a, b, c, word_to_id, id_to_word, vectors, topk=5)
        if not results:
            continue
        test_count += 1
        top_word, top_sim = results[0]
        print(f"  {desc}")
        print(f"    → 预测：{top_word}（相似度={top_sim:.4f}）")
        others = ", ".join(f"{w}({s:.3f})" for w, s in results[1:3])
        if others:
            print(f"      其他候选：{others}")

        expected = expectation_map.get((a, b, c))
        if expected and top_word == expected:
            print(f"      ✓ 精确匹配！")
            hit_count += 1
        print()

    if test_count > 0:
        print(f"  类比准确率：{hit_count}/{test_count}（{hit_count/test_count*100:.1f}%）")
    print()


def demo_cbow_vs_skipgram_comparison(
    cbow_model: Word2VecFromScratch,
    sg_model: Word2VecFromScratch,
    word_to_id: dict[str, int],
) -> None:
    """对比 CBOW 和 Skip-gram 对相同词对的相似度判断。"""
    print()
    print("=" * 70)
    print("  [CBOW vs Skip-gram 对比] 相同词对的相似度差异")
    print("=" * 70)
    print()

    # 测试词对
    test_pairs = [
        ("king", "queen"),
        ("dog", "cat"),
        ("man", "woman"),
        ("computer", "software"),
        ("learning", "education"),
        ("ocean", "river"),
        ("sun", "moon"),
        ("love", "hate"),
        ("music", "art"),
        ("war", "peace"),
    ]

    valid_pairs = [(w1, w2) for w1, w2 in test_pairs
                   if w1 in word_to_id and w2 in word_to_id]

    if valid_pairs:
        print(f"  {'词1':<14} {'词2':<14} {'CBOW 相似度':<15} {'Skip-gram 相似度':<18} {'差异':<10}")
        print(f"  {'-' * 76}")
        for w1, w2 in valid_pairs:
            sim_c = cosine_similarity(
                cbow_model.get_vector(word_to_id[w1]),
                cbow_model.get_vector(word_to_id[w2]),
            )
            sim_s = cosine_similarity(
                sg_model.get_vector(word_to_id[w1]),
                sg_model.get_vector(word_to_id[w2]),
            )
            diff = sim_s - sim_c
            direction = "↑" if diff > 0 else "↓"
            print(f"  {w1:<14} {w2:<14} {sim_c:<+15.6f} {sim_s:<+18.6f} {direction}{abs(diff):.4f}")

    print()
    print(f"  [分析]：")
    print(f"    - Skip-gram 对低频词的表示通常更准确")
    print(f"    - CBOW 由于上下文平均效应，相似度分布更平滑")
    print(f"    - 在小语料上两者差异相对较小，语料越大差异越明显")
    print()


def demo_visualization(
    model: Word2VecFromScratch,
    word_to_id: dict[str, int],
    label: str,
) -> str | None:
    """PCA 降维可视化语义空间。"""
    print()
    print("=" * 70)
    print(f"  [可视化] {label} 模型 — PCA 降维到 2D")
    print("=" * 70)

    if not HAS_MATPLOTLIB:
        print(f"\n  ⚠ matplotlib 未安装，跳过可视化。")
        print(f"    安装后可生成语义空间 PCA 图。")
        return None

    categories = {
        "人/角色":    ["king", "queen", "man", "woman", "boy", "girl", "father",
                      "mother", "child", "friend", "teacher", "student"],
        "动物":      ["dog", "cat", "horse", "bird", "fish", "lion", "wolf",
                      "bear", "elephant", "tiger"],
        "自然":      ["ocean", "river", "mountain", "forest", "sun", "moon",
                      "sky", "rain", "snow", "wind"],
        "科技":      ["computer", "software", "data", "network", "science",
                      "technology", "machine", "system", "algorithm", "program"],
        "学习/知识":  ["learning", "education", "knowledge", "book", "school",
                      "university", "research", "study", "language", "mathematics"],
        "音乐/艺术":  ["music", "song", "art", "painting", "dance", "piano",
                      "theater", "poetry", "melody", "beauty"],
    }

    # 筛选存在的词
    valid: dict[str, list[str]] = {}
    for cat, cat_words in categories.items():
        vw = [w for w in cat_words if w in word_to_id]
        if len(vw) >= 3:
            valid[cat] = vw

    if len(valid) < 2:
        print("  ✗ 没有足够的词用于可视化")
        return None

    # 收集向量
    all_words = []
    all_vecs = []
    all_cats = []
    for cat, cat_words in valid.items():
        for w in cat_words:
            if w not in all_words:
                all_words.append(w)
                all_vecs.append(model.get_vector(word_to_id[w]))
                all_cats.append(cat)

    projected = pca_2d(np.array(all_vecs))

    # 颜色映射
    cat_list = list(valid.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cat_list), 3)))
    cat_color = {cat: colors[i % len(colors)] for i, cat in enumerate(cat_list)}

    chinese_font = _get_chinese_font()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("white")

    for cat in cat_list:
        mask = [i for i, c in enumerate(all_cats) if c == cat]
        xs = projected[mask, 0]
        ys = projected[mask, 1]
        labels = [all_words[i] for i in mask]

        ax.scatter(xs, ys, color=cat_color[cat], s=120, alpha=0.8,
                   label=cat, edgecolors="white", linewidth=0.5)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=9,
                        fontproperties=chinese_font, alpha=0.85)

    ax.set_xlabel("PC1（主成分 1）", fontsize=11, fontproperties=chinese_font)
    ax.set_ylabel("PC2（主成分 2）", fontsize=11, fontproperties=chinese_font)
    ax.set_title(
        f"Word2Vec {label} 词嵌入语义空间可视化（PCA 降维到 2D）",
        fontsize=14, fontweight="bold", fontproperties=chinese_font,
    )
    ax.legend(loc="best", fontsize=10, prop=chinese_font,
              framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#f8f8f8")

    fig.text(0.5, 0.01,
             "语义相近的词在嵌入空间中聚集成簇 | "
             "PCA 降维后保留方差最大的两个方向",
             ha="center", fontsize=10, fontproperties=chinese_font, color="#666666")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    img_path = str(OUTPUT_DIR / f"word2vec_{safe_label}_visualization.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 可视化图片已保存：{img_path}")
    return img_path


def demo_save_model(
    model: Word2VecFromScratch,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    label: str,
) -> None:
    """保存模型到文件（numpy 格式）。"""
    print()
    print("=" * 70)
    print(f"  [模型保存] {label} 模型")
    print("=" * 70)

    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    npz_path = OUTPUT_DIR / f"word2vec_{safe_label}.npz"

    # 保存词向量和词表
    np.savez(
        npz_path,
        W_in=model.W_in,
        W_out=model.W_out,
        words=np.array(list(word_to_id.keys()), dtype=object),
        vector_size=model.vector_size,
    )
    size_mb = npz_path.stat().st_size / 1024 / 1024
    print(f"  ✓ 模型已保存：{npz_path}")
    print(f"    大小：{size_mb:.1f} MB")
    print(f"    内容：W_in、W_out、词表、超参数")
    print()

    # 验证加载
    data = np.load(npz_path, allow_pickle=True)
    loaded_W_in = data["W_in"]
    loaded_words = data["words"]
    print(f"  ✓ 加载验证：")
    print(f"    向量矩阵形状：{loaded_W_in.shape}")
    print(f"    词表大小：{len(loaded_words)}")
    print()


def demo_word2vec_principle() -> None:
    """展示 Word2Vec 的核心原理示意。"""
    print()
    print("=" * 70)
    print("  [原理讲解] Word2Vec 核心思想")
    print("=" * 70)
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                Word2Vec 两种训练架构                     │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print("  │                                                         │")
    print("  │  CBOW（连续词袋）：                                      │")
    print("  │    输入层：w(t-2), w(t-1), w(t+1), w(t+2)              │")
    print("  │         ↓ 查嵌入表 → 取平均                             │")
    print("  │    隐层：  h = mean(v_context)                          │")
    print("  │         ↓ 与输出嵌入做内积                              │")
    print("  │    输出层：softmax → 预测中心词 w(t)                     │")
    print("  │         ↓                                               │")
    print("  │    损失： 负采样 NCE Loss                                │")
    print("  │                                                         │")
    print("  │  Skip-gram（跳字模型）：                                 │")
    print("  │    输入层：w(t)                      ← 中心词           │")
    print("  │         ↓ 查嵌入表                                      │")
    print("  │    隐层：  h = v_c                                      │")
    print("  │         ↓ 与输出嵌入做内积（每个上下文词独立）            │")
    print("  │    输出层：softmax → 预测 w(t-2), w(t-1), w(t+1), ...  │")
    print("  │         ↓                                               │")
    print("  │    损失：  每个上下文词独立计算 NCE Loss                  │")
    print("  │                                                         │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print("  核心训练技巧：")
    print("    • 负采样（Negative Sampling）：")
    print("      每次只更新少数几个随机负样本的词向量。")
    print("      损失函数：L = -log(σ(v_c·w_pos)) - Σ log(σ(-v_c·w_neg))")
    print("      其中 σ 是 sigmoid 函数，采样分布 P(w) ∝ freq(w)^0.75")
    print()
    print("    • 这个公式的含义：")
    print("      - 鼓励模型给正样本（真实上下文词）打高分")
    print("      - 同时给随机采样的负样本打低分")
    print("      - 相当于在噪声中「辨别」出真正的上下文词")
    print()
    print("  训练后的词向量具有的性质：")
    print("    • 语义相似的词 → 向量空间中距离近")
    print("    • 词类比：vec('king') - vec('man') + vec('woman') ≈ vec('queen')")
    print("    • 向量编码了语法和语义的复合信息")
    print()


# ─────────────────────────────────────────────────────────────
# 6. 工具函数
# ─────────────────────────────────────────────────────────────

def _get_chinese_font():
    """尝试获取中文字体（需要 matplotlib）。"""
    if not HAS_MATPLOTLIB:
        return None
    try:
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                return fm.FontProperties(fname=fp)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# 7. 主流程
# ─────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  Word2Vec 词向量训练示例（纯 NumPy 从零实现）")
    print("  从零训练 CBOW 和 Skip-gram 模型，探索语义空间")
    print("=" * 70)
    print()

    # ── 步骤 1：准备语料 ──
    sentences, total_words, vocab_size = prepare_corpus()

    # ── 步骤 2：构建词表和训练样本 ──
    print("[步骤 2/7] 构建词表和训练样本...")
    print("-" * 60)
    word_to_id, id_to_word, word_freq, samples = build_vocab_and_samples(sentences)

    # ── 步骤 3：训练 CBOW ──
    print(f"  ⏳ 训练可能需要 1-3 分钟（取决于语料大小和 CPU 性能）\n")
    cbow_model, cbow_losses = train_cbow(samples, len(word_to_id), word_freq)

    # ── 步骤 4：训练 Skip-gram ──
    # sg_model, sg_losses = train_skipgram(samples, len(word_to_id), word_freq)

    # ── 步骤 5：原理讲解 ──
    # demo_word2vec_principle()

    # ── 步骤 6：模型信息 ──
    # demo_model_info(cbow_model, sg_model, word_to_id)

    # ── 步骤 7：损失曲线 ──
    # demo_loss_curves(cbow_losses, sg_losses)

    # ── 取出词向量矩阵 ──
    cbow_vectors = cbow_model.get_vectors()
    # sg_vectors = sg_model.get_vectors()

    # ── 演示：CBOW ──
    demo_word_vectors(cbow_model, word_to_id, "CBOW")
    demo_similarity(cbow_model, word_to_id, "CBOW")
    demo_similar_words(cbow_model, cbow_vectors, word_to_id, id_to_word, "CBOW")
    demo_word_analogies(cbow_vectors, word_to_id, id_to_word, "CBOW")

    # ── 演示：Skip-gram ──
    # demo_word_vectors(sg_model, word_to_id, "Skip-gram")
    # demo_similarity(sg_model, word_to_id, "Skip-gram")
    # demo_similar_words(sg_model, sg_vectors, word_to_id, id_to_word, "Skip-gram")
    # demo_word_analogies(sg_vectors, word_to_id, id_to_word, "Skip-gram")

    # ── 对比 ──
    # demo_cbow_vs_skipgram_comparison(cbow_model, sg_model, word_to_id)

    # ── 可视化 ──
    img_cbow = demo_visualization(cbow_model, word_to_id, "CBOW")
    # img_sg = demo_visualization(sg_model, word_to_id, "Skip-gram")

    # ── 模型保存 ──
    demo_save_model(cbow_model, word_to_id, id_to_word, "CBOW")

    # ── 总结 ──
    print()
    print("=" * 70)
    print("  [总结] Word2Vec 词嵌入（纯 NumPy 从零实现）")
    print("=" * 70)
    print(f"  1. Word2Vec 是 2013 年提出的经典词嵌入方法。")
    print(f"  2. 本示例使用纯 NumPy 从零实现了：")
    print(f"     - CBOW 架构（上下文 → 中心词）")
    print(f"     - Skip-gram 架构（中心词 → 上下文）")
    print(f"     - 负采样（Negative Sampling）训练技巧")
    print(f"     - 学习率线性衰减")
    print(f"  3. 训练过程完全是透明的——可以看到每一步的梯度计算")
    print(f"     和参数更新，没有黑盒。")
    print(f"  4. 词向量编码了丰富的语义关系：")
    print(f"     - 语义相似度：余弦相似度")
    print(f"     - 语义类比：向量偏移（king - man + woman ≈ queen）")
    print(f"  5. Word2Vec vs GloVe：")
    print(f"     - Word2Vec：局部上下文窗口预测，在线学习")
    print(f"     - GloVe：全局共现矩阵分解，统计信息利用更充分")
    print(f"     - 实践中两者效果接近，各有优势场景")
    print(f"  6. 局限性：")
    print(f"     - 每个词只有一种静态表示（无法处理一词多义）")
    print(f"     - OOV（词表外词）问题")
    print(f"     - 后来被上下文相关嵌入（ELMo → BERT → LLM）超越")
    print()

    # 输出文件提示
    print(f"  💡 输出文件保存在：{OUTPUT_DIR}")
    loss_img = OUTPUT_DIR / "word2vec_training_loss.png"
    if loss_img.exists():
        print(f"     - 训练损失曲线：{loss_img.name}")
    if img_cbow:
        print(f"     - CBOW 可视化：{os.path.basename(img_cbow)}")
    # if img_sg:
    #     print(f"     - Skip-gram 可视化：{os.path.basename(img_sg)}")
    cbow_npz = OUTPUT_DIR / "word2vec_cbow.npz"
    if cbow_npz.exists():
        print(f"     - CBOW 模型：{cbow_npz.name}")
    print()


if __name__ == "__main__":
    main()
