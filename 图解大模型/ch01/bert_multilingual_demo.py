"""
bert-base-multilingual-cased 演示 — 多语言 BERT 基础功能

BERT (Bidirectional Encoder Representations from Transformers) 是 2018 年
Google 提出的预训练模型，核心创新：
  - 双向上下文：同时看左右两侧的词语来理解语义
  - Masked Language Model (MLM)：随机遮住词，让模型预测被遮住的词
  - Next Sentence Prediction (NSP)：判断两句话是否连续

模型配置（bert-base-multilingual-cased）：
  - 层数：12 层 Transformer
  - 隐藏层维度：768
  - 注意力头数：12
  - 参数量：~110M
  - 支持 104 种语言（含中文、英文、日文、法文等）
  - 词表大小：119,547（WordPiece 子词）

本示例：
  1. 加载模型和分词器
  2. 分词演示（WordPiece 子词拆分）
  3. 提取文本嵌入向量
  4. 多语言语义相似度对比
  5. Masked Language Model 推理（模型填空）

首次运行会自动从 hf-mirror.com 下载约 700MB 模型文件。
"""

import os
import sys
import time
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ── 设置 HuggingFace 镜像源 ─────────────────────────────────────
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


# ── 常量 ─────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-multilingual-cased"

# 用于 MLM 填空的句子
MASK_SENTENCE = "I love [MASK] learning."        # 中文：Paris is a beautiful [MASK].
MASK_SENTENCE_ZH = "我热爱[MASK]学习。"

# 用于多语言嵌入比较的句子 (相同语义，不同语言)
SENTENCE_EN = "I love deep learning"
SENTENCE_ZH = "我热爱深度学习"
SENTENCE_JP = "私は深層学習が大好きです"
SENTENCE_FR = "J'adore l'apprentissage profond"

# ── 模型加载 ────────────────────────────────────────────────────


def load_model_and_tokenizer():
    """加载 bert-base-multilingual-cased 模型和分词器。"""
    print("=" * 70)
    print(f"  模型：{MODEL_NAME}")
    print(f"  PyTorch: {torch.__version__} (CPU)")
    print(f"  类型：多语言 BERT（支持 104 种语言）")
    print(f"  参数量：≈110M")
    print(f"  大小：约 700 MB")
    print("=" * 70)
    print()

    print("[1/2] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  OK | vocab_size: {tokenizer.vocab_size:,}")
    print(f"  OK | pad_token: {tokenizer.pad_token} (ID:{tokenizer.pad_token_id})")
    print(f"  OK | cls_token: {tokenizer.cls_token} (ID:{tokenizer.cls_token_id})")
    print(f"  OK | sep_token: {tokenizer.sep_token} (ID:{tokenizer.sep_token_id})")
    print(f"  OK | mask_token: {tokenizer.mask_token} (ID:{tokenizer.mask_token_id})")
    print(f"  OK | unk_token: {tokenizer.unk_token} (ID:{tokenizer.unk_token_id})")
    print()

    print("[2/2] 加载模型（首次会下载约 700MB）...")
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK | 参数量：{total_params / 1e6:.0f}M")
    print(f"  OK | 耗时：{time.time() - t0:.1f} 秒")
    print()

    # 打印模型结构的前几层
    print("模型结构预览（仅顶层模块）：")
    print(model)
    print()

    return model, tokenizer


# ── 1. 分词演示 ────────────────────────────────────────────────


def demo_tokenization(tokenizer):
    """展示 WordPiece 分词器如何将不同语言的文本拆分为子词。"""
    print("=" * 70)
    print("  [1/5] 分词演示 — WordPiece 子词拆分")
    print("=" * 70)
    print()
    print("  BERT 使用 WordPiece 分词器：")
    print("  - 常见词保持完整（如 'love'）")
    print("  - 生僻词或词形变化拆分为子词（如 'unhappiness' → un + ##happiness）")
    print("  - ## 标记表示该子词是某个词的中间或结尾部分")
    print()

    texts = [
        "I love deep learning",
        "unhappiness",
        "Transformers are powerful",
        "我热爱深度学习",
        "私は深層学習が大好きです",
    ]

    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        # 用特殊标记 [CLS] 和 [SEP] 加上首尾
        input_ids = tokenizer.encode(text)

        tokens_label = tokenizer.convert_ids_to_tokens(input_ids)

        print(f"  原文：{text}")
        print(f"  分词（{len(tokens)} 个子词）：{tokens}")
        print(f"  ID：{ids}")
        print(f"  带 [CLS]/[SEP]：{tokens_label}")
        print()

    # 展示子词拆分的细节
    print("  ── 子词拆分细节 ──")
    detail_word = "unhappiness"
    tokens = tokenizer.tokenize(detail_word)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"  '{detail_word}' 拆分为 {len(tokens)} 个子词：")
    for i, (tok, tid) in enumerate(zip(tokens, ids)):
        note = "完整词" if not tok.startswith("##") else "续词片段"
        print(f"    [{i}] {tok:<12} ID:{tid:<6} ({note})")

    print()
    print("  💡 WordPiece 的核心思路：用有限的词表覆盖无限的词汇。")
    print("     常见词保留完整，生僻词拆成已知子词，永不出现 <UNK>。")
    print()


# ── 2. 嵌入向量演示 ────────────────────────────────────────────


def demo_embeddings(model, tokenizer):
    """提取文本的 BERT 嵌入向量（CLS 向量 + 均值池化）。"""
    print("=" * 70)
    print("  [2/5] 嵌入向量 — 文本的语义表示")
    print("=" * 70)
    print()
    print("  BERT 为每个 token 输出一个 768 维的隐藏状态向量。")
    print("  常用两种方式获得句子级嵌入：")
    print("    a) [CLS] 向量：BERT 用 [CLS] 位置的输出聚合整句信息")
    print("    b) 均值池化：对所有 token 的输出取平均")
    print()

    text = "I love deep learning"
    print(f"  输入文本：{text}")
    print()

    # 编码并前向传播
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, 768)
    seq_len, hidden_dim = last_hidden.shape[1], last_hidden.shape[2]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    print(f"  隐藏状态维度：{hidden_dim}")
    print(f"  序列长度（含特殊标记）：{seq_len}")
    print()

    # 展示每个 token 的嵌入向量（前 6 维）
    print(f"  各 Token 的嵌入向量（显示前 6 维，总 {hidden_dim} 维）：")
    for i, token in enumerate(tokens):
        vec = last_hidden[0, i, :6].tolist()
        fmt = ", ".join(f"{v:+.4f}" for v in vec)
        marker = " ← [CLS]（句子级表示）" if i == 0 else " ← [SEP]" if i == seq_len - 1 else ""
        print(f"    [{i}] \"{token}\"  →  [{fmt}, ...]   {marker}")

    # [CLS] 向量
    cls_vec = last_hidden[0, 0, :]  # (768,)
    print()
    print(f"  ▸ [CLS] 向量（前 10 维）：")
    cls_front = ", ".join(f"{v:+.4f}" for v in cls_vec[:10].tolist())
    print(f"    [{cls_front}, ...]  (共 {hidden_dim} 维)")
    print(f"    L2 范数：{torch.norm(cls_vec, p=2).item():.4f}")

    # 均值池化向量
    mean_vec = last_hidden[0, 1:-1, :].mean(dim=0)  # 除掉 [CLS] 和 [SEP]
    print()
    print(f"  ▸ 均值池化向量（去除 [CLS]/[SEP] 后取平均，前 10 维）：")
    mean_front = ", ".join(f"{v:+.4f}" for v in mean_vec[:10].tolist())
    print(f"    [{mean_front}, ...]  (共 {hidden_dim} 维)")
    print(f"    L2 范数：{torch.norm(mean_vec, p=2).item():.4f}")

    print()
    print(f"  💡 [CLS] 向量是 BERT 为分类任务设计的句子表示，")
    print(f"     在 MLM 预训练中 [CLS] 也能捕获整句信息。")
    print()


# ── 3. 多语言语义相似度 ────────────────────────────────────────


def demo_multilingual_similarity(model, tokenizer):
    """对比同一语义用不同语言表达时的嵌入相似度。"""
    print("=" * 70)
    print("  [3/5] 多语言语义相似度 — 跨语言嵌入")
    print("=" * 70)
    print()
    print("  bert-base-multilingual-cased 在 104 种语言的 Wikipedia 上训练，")
    print("  因此不同语言的相同语义在嵌入空间中应该距离较近。")
    print()

    sentences = {
        "英文": SENTENCE_EN,
        "中文": SENTENCE_ZH,
        "日文": SENTENCE_JP,
        "法文": SENTENCE_FR,
    }

    # 提取各语言的句子嵌入
    embeddings = {}
    with torch.no_grad():
        for lang, text in sentences.items():
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            # 使用均值池化（去除 [CLS]/[SEP]）
            emb = outputs.hidden_states[-1][0, 1:-1, :].mean(dim=0)
            embeddings[lang] = F.normalize(emb, dim=0)

    print(f"  句子含义：\"{SENTENCE_EN}\"")
    print()

    # 计算余弦相似度矩阵
    print(f"  {'':<8} {'英文':<12} {'中文':<12} {'日文':<12} {'法文':<12}")
    print(f"  {'-' * 56}")
    langs = list(sentences.keys())
    for lang1 in langs:
        row = f"  {lang1:<6}"
        for lang2 in langs:
            sim = (embeddings[lang1] @ embeddings[lang2]).item()
            row += f" {sim:<+11.4f} "
        print(row)

    print()
    print("  💡 跨语言语义相似度越高，说明模型的多语言\"对齐\"效果越好。")
    print("     同一句话的不同语言版本在嵌入空间中应彼此靠近。")

    # 额外演示：不同语义句子的区分
    print()
    print("  ── 对比：不同语义句子的相似度（应当低于同语义） ──")
    other_texts = {
        "英文-不相关": "The weather is nice today",
        "中文-不相关": "今天天气真好",
    }

    other_embs = {}
    with torch.no_grad():
        for label, text in other_texts.items():
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            emb = outputs.hidden_states[-1][0, 1:-1, :].mean(dim=0)
            other_embs[label] = F.normalize(emb, dim=0)

    for label, emb in other_embs.items():
        sim_en = (embeddings["英文"] @ emb).item()
        sim_zh = (embeddings["中文"] @ emb).item()
        print(f"  {label:<20} vs 英文: {sim_en:.4f}  vs 中文: {sim_zh:.4f}")

    print()
    print("  对比结论：同语义跨语言相似度通常 > 异语义相似度，")
    print("  说明 BERT 的多语言嵌入具有跨语言泛化能力。")
    print()


# ── 4. Masked Language Model ────────────────────────────────────


def demo_masked_lm(model, tokenizer):
    """
    Masked Language Model 演示。

    BERT 在训练时随机遮盖输入文本中 15% 的 token，
    然后让模型根据双向上下文预测被遮住的词。
    这是 BERT 最重要的自监督训练任务。
    """
    print("=" * 70)
    print("  [4/5] Masked Language Model — 根据上下文预测遮住的词")
    print("=" * 70)
    print()
    print("  这是 BERT 的核心预训练任务，类似完形填空。")
    print("  模型同时看左右两侧的上下文来预测 [MASK] 位置的词。")
    print()

    # ── 英文填空 ──
    print(f"  英文句子：\"{MASK_SENTENCE}\"")
    inputs_en = tokenizer(MASK_SENTENCE, return_tensors="pt")
    mask_idx_en = torch.where(inputs_en["input_ids"][0] == tokenizer.mask_token_id)[0].item()

    with torch.no_grad():
        outputs_en = model(**inputs_en)
        logits_en = outputs_en.logits[0, mask_idx_en]  # (vocab_size,)

    probs_en = F.softmax(logits_en, dim=-1)
    top_probs_en, top_indices_en = torch.topk(probs_en, k=10)

    print(f"  [MASK] 位置：{mask_idx_en}")
    print(f"  预测 Top-10：")
    print(f"  {'#':<4} {'token':<20} {'概率':<14}")
    print(f"  {'-' * 40}")
    for i, (prob, idx) in enumerate(zip(top_probs_en.tolist(), top_indices_en.tolist())):
        token_str = tokenizer.convert_ids_to_tokens(idx)
        print(f"  {i+1:<4} {token_str:<20} {prob:<+14.2%}")
    print()

    # ── 中文填空 ──
    print(f"  中文句子：\"{MASK_SENTENCE_ZH}\"")
    inputs_zh = tokenizer(MASK_SENTENCE_ZH, return_tensors="pt")
    mask_idx_zh = torch.where(inputs_zh["input_ids"][0] == tokenizer.mask_token_id)[0].item()

    with torch.no_grad():
        outputs_zh = model(**inputs_zh)
        logits_zh = outputs_zh.logits[0, mask_idx_zh]

    probs_zh = F.softmax(logits_zh, dim=-1)
    top_probs_zh, top_indices_zh = torch.topk(probs_zh, k=10)

    print(f"  [MASK] 位置：{mask_idx_zh}")
    print(f"  预测 Top-10：")
    print(f"  {'#':<4} {'token':<20} {'概率':<14}")
    print(f"  {'-' * 40}")
    for i, (prob, idx) in enumerate(zip(top_probs_zh.tolist(), top_indices_zh.tolist())):
        token_str = tokenizer.convert_ids_to_tokens(idx)
        print(f"  {i+1:<4} {token_str:<20} {prob:<+14.2%}")
    print()

    # ── 构造完整的填空结果 ──
    def fill_mask(sentence: str) -> str:
        """用 Top-1 预测填充 [MASK] 并返回完整句子。"""
        inputs = tokenizer(sentence, return_tensors="pt")
        mask_idx = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()
        with torch.no_grad():
            logits = model(**inputs).logits[0, mask_idx]
        top_id = torch.argmax(logits).item()
        predicted = tokenizer.convert_ids_to_tokens(top_id)

        filled = sentence.replace("[MASK]", predicted, 1)
        # 清理 WordPiece ## 连接
        if predicted.startswith("##"):
            filled = sentence.replace("[MASK]", predicted[2:], 1)
        return filled, predicted

    filled_en, pred_en = fill_mask(MASK_SENTENCE)
    filled_zh, pred_zh = fill_mask(MASK_SENTENCE_ZH)

    print(f"  ── 填空结果 ──")
    print(f"  英文：\"{MASK_SENTENCE}\"  →  \"{filled_en}\"  (预测: {pred_en})")
    print(f"  中文：\"{MASK_SENTENCE_ZH}\"  →  \"{filled_zh}\"  (预测: {pred_zh})")
    print()

    print("  💡 注意 BERT 的 MLM 预测基于双向上下文。")
    print("     这不同于 GPT 系列的自回归方式（只能看左侧）。")
    print("     双向上下文使 BERT 在理解类任务上表现更好。")
    print()


# ── 5. 模型配置与统计信息 ─────────────────────────────────────


def demo_model_info(model, tokenizer):
    """展示 BERT 模型的配置信息和统计。"""
    print("=" * 70)
    print("  [5/5] 模型配置与参数统计")
    print("=" * 70)
    print()

    config = model.config
    print(f"  ── 模型配置 ──")
    print(f"  架构：{config.architectures[0] if hasattr(config, 'architectures') and config.architectures else 'BERT'}")
    print(f"  隐藏层维度 (hidden_size)：{config.hidden_size}")
    print(f"  Transformer 层数 (num_hidden_layers)：{config.num_hidden_layers}")
    print(f"  注意力头数 (num_attention_heads)：{config.num_attention_heads}")
    print(f"  每头维度：{config.hidden_size // config.num_attention_heads}")
    print(f"  中间层维度 (intermediate_size)：{config.intermediate_size}")
    print(f"  词表大小 (vocab_size)：{config.vocab_size:,}")
    print(f"  最大位置编码 (max_position_embeddings)：{config.max_position_embeddings}")
    print(f"  激活函数：{config.hidden_act}")
    print(f"  Dropout 概率：{config.hidden_dropout_prob}")
    print(f"  注意力 Dropout：{config.attention_probs_dropout_prob}")
    print()

    # 参数统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if "embedding" in n)
    encoder_params = total - embedding_params

    print(f"  ── 参数统计 ──")
    print(f"  总参数量：{total:,} ({total/1e6:.1f}M)")
    print(f"  可训练参数量：{trainable:,} ({trainable/1e6:.1f}M)")
    print(f"  嵌入层参数：{embedding_params:,} ({embedding_params/1e6:.1f}M) "
          f"({embedding_params/total*100:.0f}%)")
    print(f"  Transformer 编码器参数：{encoder_params:,} ({encoder_params/1e6:.1f}M) "
          f"({encoder_params/total*100:.0f}%)")
    print()

    # 各层参数分布
    print(f"  ── 每层参数分布 ──")
    layer_params = {}
    for name, param in model.named_parameters():
        # 提取层编号
        parts = name.split(".")
        layer_key = "embedding" if "embedding" in parts else "pooler"
        for p in parts:
            if p.startswith("layer"):
                layer_key = p
                break
        layer_params[layer_key] = layer_params.get(layer_key, 0) + param.numel()

    print(f"  {'模块':<20} {'参数量':<12} {'占比':<10}")
    print(f"  {'-' * 42}")
    for key in sorted(layer_params.keys(), key=lambda k: (
        0 if k == "embedding" else (1 if k.startswith("layer") else 2), k
    )):
        count = layer_params[key]
        pct = count / total * 100
        label = f"embedding (0)" if key == "embedding" else key if key.startswith("layer.") else key
        print(f"  {label:<20} {count:<12,} {pct:<10.1f}%")

    print()
    print(f"  ── BERT 模型特点总结 ──")
    print(f"  1. 编码器架构：只有 Transformer Encoder（无 Decoder）")
    print(f"  2. 双向注意力：每个位置都能看到所有位置（非因果）")
    print(f"  3. 预训练任务：MLM（掩码语言模型）+ NSP（下一句预测）")
    print(f"  4. 输入组成：[CLS] + token IDs + [SEP] + segment IDs + position IDs")
    print(f"  5. 输出：每个 token 对应的隐藏状态 + 池化后的 [CLS] 向量")
    print(f"  6. 多语言：104 语言联合训练，共享词表和嵌入空间")
    print()


# ── 主函数 ─────────────────────────────────────────────────────


def main():
    print()
    print("=" * 70)
    print("   bert-base-multilingual-cased 演示")
    print("   多语言 BERT 基础功能 — 分词 / 嵌入 / MLM")
    print("=" * 70)
    print()

    model, tokenizer = load_model_and_tokenizer()

    demo_tokenization(tokenizer)
    demo_embeddings(model, tokenizer)
    demo_multilingual_similarity(model, tokenizer)
    demo_masked_lm(model, tokenizer)
    demo_model_info(model, tokenizer)

    # ── 总结 ──
    print("=" * 70)
    print("  [总结] BERT 模型的核心贡献")
    print("=" * 70)
    print()
    print("  1. 双向上下文编码 — 突破了传统语言模型仅单向建模的限制")
    print("  2. MLM 预训练范式 — 模型能够深度理解上下文语义关系")
    print("  3. 统一的微调框架 — 在多种 NLP 任务上只需加一个分类头")
    print("  4. 多语言泛化 — 跨语言共享语义空间，支持零样本迁移")
    print()
    print("  从 GloVe 到 BERT 的演进：")
    print("    GloVe (2014) → 静态词向量，一词一义")
    print("    ELMo  (2018) → 上下文相关的双向 LSTM 嵌入")
    print("    BERT  (2018) → Transformer 双向编码，MLM 预训练")
    print("    GPT   (2018) → Transformer 自回归解码，生成式预训练")
    print()
    print(f"  本演示使用的模型：bert-base-multilingual-cased (~110M)")
    print(f"  数据来源：hf-mirror.com（HuggingFace 国内镜像）")
    print()


if __name__ == "__main__":
    main()
