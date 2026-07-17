"""
bert-base-multilingual-cased 电影评价分类演示

用 BERT 的 [CLS] 嵌入向量对电影评价做情感分类（无需微调）。
核心思路：
  1. 用预训练 BERT 提取每条评价的语义嵌入向量
  2. 在嵌入空间中计算评价之间的余弦相似度
  3. 通过最近邻方法预测新评价的情感倾向（正面/负面）

模型和数据均从 hf-mirror.com 下载。
"""

import os
import sys
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ── 带标签的电影评价（中英文混合） ────────────────────────────

LABELED_REVIEWS = [
    # 正面评价
    ("This movie is absolutely fantastic, the best I have ever seen!", "positive"),
    ("演技精湛，剧情引人入胜，非常推荐", "positive"),
    ("Brilliant cinematography and a touching story. A masterpiece!", "positive"),
    ("导演的手法太棒了，每个镜头都充满艺术感", "positive"),
    ("An inspiring film with incredible performances from the cast.", "positive"),
    ("画面精美，配乐动听，今年最好的电影", "positive"),

    # 负面评价
    ("Terrible movie, wasted two hours of my life.", "negative"),
    ("剧情无聊透顶，演员演技尴尬", "negative"),
    ("The plot made no sense and the dialogue was cringeworthy.", "negative"),
    ("特效粗糙，故事也毫无新意，浪费时间", "negative"),
    ("Disjointed storytelling and wooden acting. Avoid at all costs.", "negative"),
    ("剪辑混乱，完全看不懂在讲什么", "negative"),
]

# 待预测的新评价（无标签）
UNLABELED_REVIEWS = [
    "An absolutely wonderful film that touched my heart deeply.",
    "这个电影太难看了，我看了半小时就睡着了",
    "不怎么样，一般般吧",
    "A decent movie with some nice moments, but nothing special.",
    "The worst film I have seen this year. Complete disaster.",
    "演技在线，故事也很感人，值得一看",
    "特效不错但剧情太弱，总体还行",
    "Brilliant from start to finish. A true cinematic achievement.",
]


# ── 模型加载 ────────────────────────────────────────────────────

def load_model():
    print("=" * 70)
    print("  bert-base-multilingual-cased 电影评价分类器")
    print("  基于 [CLS] 嵌入向量 + 最近邻的零样本分类")
    print("=" * 70)
    print()

    print("[1/3] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    print(f"  OK | vocab_size: {tokenizer.vocab_size:,}")

    print("[2/3] 加载模型（首次自动下载约 700MB）...")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK | 参数量：{total_params / 1e6:.0f}M")
    print()

    return model, tokenizer


# ── 计算句子嵌入向量 ────────────────────────────────────────────

@torch.no_grad()
def get_sentence_embedding(model, tokenizer, text: str) -> torch.Tensor:
    """
    用 BERT 提取句子的 [CLS] 嵌入向量。
    [CLS] 是 BERT 设计中用于聚合整句信息的特殊标记。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    # 取 [CLS] 位置的输出（第一个 token）作为句子表示
    cls_vec = outputs.last_hidden_state[0, 0]  # (768,)
    return F.normalize(cls_vec, dim=0)  # L2 归一化，方便直接点积算余弦相似度


@torch.no_grad()
def compute_all_embeddings(model, tokenizer, reviews: list[str]) -> torch.Tensor:
    """批量计算所有评级的嵌入向量。"""
    embs = []
    for text in reviews:
        vec = get_sentence_embedding(model, tokenizer, text)
        embs.append(vec)
    return torch.stack(embs)


# ── 构建分类器 ────────────────────────────────────────────────────

def build_classifier(model, tokenizer):
    """
    用带标签的评价构建嵌入向量库。
    返回：
      label_embs : 形状 (N, 768)，已归一化
      labels     : 对应的标签列表
      texts      : 对应的原文列表
    """
    texts = [r[0] for r in LABELED_REVIEWS]
    labels = [r[1] for r in LABELED_REVIEWS]
    label_embs = compute_all_embeddings(model, tokenizer, texts)
    return label_embs, labels, texts


def predict(text: str, label_embs: torch.Tensor, labels: list[str],
            texts: list[str], k: int = 3) -> tuple[str, float, list]:
    """
    对一条新评价做分类。

    步骤：
      1. 提取新评价的嵌入向量
      2. 计算与所有已知评价的余弦相似度（点积，因为都已归一化）
      3. 取最相似的 k 条，按多数投票决定情感倾向

    返回：
      (预测标签, 置信度, [(相似度, 标签, 原文), ...])
    """
    vec = get_sentence_embedding(model, tokenizer, text)  # (768,)
    sims = (vec @ label_embs.T).squeeze()  # (N,)

    # 取 top-k 最相似
    top_sims, top_idx = torch.topk(sims, k=min(k, len(labels)))

    neighbors = []
    pos_votes = 0
    for sim, idx in zip(top_sims.tolist(), top_idx.tolist()):
        neighbors.append((sim, labels[idx], texts[idx]))
        if labels[idx] == "positive":
            pos_votes += 1

    neg_votes = k - pos_votes
    pred = "positive" if pos_votes > neg_votes else "negative"

    # 置信度 = 胜出票数比例
    confidence = max(pos_votes, neg_votes) / k

    return pred, confidence, neighbors


# ── 演示 1：展示嵌入空间的聚类效果 ────────────────────────────────

def demo_embedding_space(model, tokenizer):
    """展示正/负面评价在嵌入空间中的相似度矩阵。"""
    print("=" * 70)
    print("  [演示一] 评价嵌入向量的相似度矩阵")
    print("=" * 70)
    print()
    print("  对比所有带标签评价的余弦相似度，观察同类/异类距离。")
    print()

    texts = [r[0] for r in LABELED_REVIEWS]
    labels = [r[1] for r in LABELED_REVIEWS]
    embs = compute_all_embeddings(model, tokenizer, texts)
    sim_matrix = embs @ embs.T  # 余弦相似度矩阵（已归一化）

    # 打印矩阵
    print(f"  {'':<34} ", end="")
    for i in range(len(texts)):
        label_short = "P" if labels[i] == "positive" else "N"
        print(f"  {label_short}{i:<3}", end="")
    print()

    for i in range(len(texts)):
        label_short = "正" if labels[i] == "positive" else "负"
        preview = texts[i][:20] + ("..." if len(texts[i]) > 20 else "")
        print(f"  [{label_short}] {preview:<28} ", end="")
        for j in range(len(texts)):
            sim = sim_matrix[i, j].item()
            # 高亮同类高相似度、异类低相似度
            same = (labels[i] == labels[j])
            if same and sim > 0.95:
                marker = "██"
            elif same and sim > 0.90:
                marker = "▓▓"
            elif same:
                marker = "▒▒"
            elif sim < 0.70:
                marker = "░░"
            else:
                marker = "  "
            print(f" {sim:<+6.3f}{marker}", end="")
        print()

    print()
    print("  ▸ ██=同类高相似度  ▓▓=同类中相似度  ▒▒=同类低相似度")
    print("  ▸ ░░=异类低相似度（理想情况）")
    print()

    # 同类 vs 异类统计
    same_sims = []
    diff_sims = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j].item()
            if labels[i] == labels[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    same_avg = sum(same_sims) / len(same_sims) if same_sims else 0
    diff_avg = sum(diff_sims) / len(diff_sims) if diff_sims else 0
    print(f"  同类平均相似度：{same_avg:.4f}")
    print(f"  异类平均相似度：{diff_avg:.4f}")
    print(f"  区分度（同类-异类）：{same_avg - diff_avg:.4f}")
    print()
    print(f"  💡 同类评价的嵌入向量更相似，说明 BERT 有效地编码了语义情感。")
    print()


# ── 演示 2：最近邻分类 ────────────────────────────────────────────

def demo_classification(model, tokenizer):
    """用最近邻方法对无标签评价进行情感分类。"""
    print("=" * 70)
    print("  [演示二] 最近邻情感分类")
    print("=" * 70)
    print()
    print("  对新评价的流程：")
    print("    1. 用 BERT 提取 [CLS] 嵌入向量")
    print("    2. 与已知评价的嵌入向量算余弦相似度")
    print("    3. 取 Top-3 最近邻，多数投票决定情感")
    print()

    label_embs, labels, texts = build_classifier(model, tokenizer)

    for review in UNLABELED_REVIEWS:
        # 预测
        pred, confidence, neighbors = predict(review, label_embs, labels, texts)

        # 显示结果
        label_str = "🎬 正面" if pred == "positive" else "👎 负面"
        conf_pct = confidence * 100

        print(f"  评价：{review}")
        print(f"  预测：{label_str}  (置信度 {conf_pct:.0f}%)")
        print(f"  Top-3 最近邻：")
        for sim, lbl, src in neighbors:
            lbl_cn = "正面" if lbl == "positive" else "负面"
            src_short = src[:30] + ("..." if len(src) > 30 else "")
            print(f"    [{lbl_cn}] (sim={sim:.4f}) {src_short}")
        print()

    print(f"  💡 无需微调，仅用 BERT 的预训练嵌入向量 + 最近邻")
    print(f"     就能对电影评价做有意义的分类。")
    print(f"     如果要更高精度，可以在此基础微调一个分类头。")
    print()


# ── 演示 3：对比不同表达方式的影响 ────────────────────────────────

def demo_expression_impact(model, tokenizer):
    """展示同样情感、不同表达方式在嵌入空间中的距离。"""
    print("=" * 70)
    print("  [演示三] 表达方式对嵌入向量的影响")
    print("=" * 70)
    print()

    # 相同情感、不同强度的评价
    positive_reviews = [
        ("强烈正面", "This is the greatest movie I have ever seen, absolutely brilliant!"),
        ("温和正面", "It's a pretty good movie, I enjoyed it."),
        ("勉强正面", "Not bad, I guess it was okay."),
    ]

    negative_reviews = [
        ("勉强负面", "It wasn't great, could have been better."),
        ("温和负面", "I didn't really like this movie, it was boring."),
        ("强烈负面", "Absolutely terrible, the worst movie ever made!"),
    ]

    print(f"  {'强度':<12} {'评价内容':<36} {'嵌入向量':<14}")
    print(f"  {'-' * 62}")

    all_reviews = positive_reviews + negative_reviews
    for intensity, text in all_reviews:
        vec = get_sentence_embedding(model, tokenizer, text)
        front = ", ".join(f"{v:+.3f}" for v in vec[:5].tolist())
        print(f"  {intensity:<10} {text:<34} [{front}, ...]")

    print()
    print(f"  ── 强度递减下的余弦相似度 ──")

    # 正面情绪内部分析
    p_vecs = [get_sentence_embedding(model, tokenizer, t) for _, t in positive_reviews]
    print(f"  ▸ 正面区间：")
    print(f"    强烈 vs 温和：{F.cosine_similarity(p_vecs[0], p_vecs[1], dim=0).item():.4f}")
    print(f"    强烈 vs 勉强：{F.cosine_similarity(p_vecs[0], p_vecs[2], dim=0).item():.4f}")

    n_vecs = [get_sentence_embedding(model, tokenizer, t) for _, t in negative_reviews]
    print(f"  ▸ 负面区间：")
    print(f"    强烈 vs 温和：{F.cosine_similarity(n_vecs[0], n_vecs[1], dim=0).item():.4f}")
    print(f"    强烈 vs 勉强：{F.cosine_similarity(n_vecs[0], n_vecs[2], dim=0).item():.4f}")

    # 跨极性对比
    print(f"  ▸ 跨极性：")
    cross = F.cosine_similarity(p_vecs[0], n_vecs[0], dim=0).item()
    print(f"    强烈正面 vs 强烈负面：{cross:.4f}")

    print()
    print(f"  💡 相同情感倾向的评价在嵌入空间中距离更近，")
    print(f"     情感越极端，嵌入向量越远离中立区域。")
    print()


# ── 主函数 ───────────────────────────────────────────────────────

def main():
    global model, tokenizer
    model, tokenizer = load_model()

    demo_embedding_space(model, tokenizer)
    demo_classification(model, tokenizer)
    demo_expression_impact(model, tokenizer)

    # 总结
    print("=" * 70)
    print("  [总结]")
    print("=" * 70)
    print()
    print("  用 BERT 做情感分类的几种方式：")
    print()
    print("  1. 嵌入向量 + 最近邻（本演示）")
    print("     ✓ 无需微调，零样本可用")
    print("     ✓ 可解释性强（展示近邻评价）")
    print("     ✗ 精度不如微调模型")
    print()
    print("  2. 嵌入向量 + 逻辑回归")
    print("     ✓ 训练极快（只训练分类头）")
    print("     ✓ 适合标注数据较少的场景")
    print()
    print("  3. 微调整个 BERT")
    print("     ✓ 精度最高")
    print("     ✗ 需要较多标注数据和 GPU 训练")
    print()
    print(f"  模型：bert-base-multilingual-cased（{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M）")
    print()


if __name__ == "__main__":
    main()
