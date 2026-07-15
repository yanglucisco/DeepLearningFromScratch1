"""
Phi-3-mini-4k-instruct 分词器示例 — 子词级 Tokenizer 原理解析

对比 jieba（基于词典的最大正向匹配分词器）：
  - Phi-3 使用的是 BPE（Byte-Pair Encoding）子词分词器
  - 词汇表大小：32,000（含特殊 token）
  - 可以将任意词拆分为子词单元，无 OOV 问题
  - 自动处理大小写、标点、空格前缀

使用前请设置国内镜像（如无法直接访问 HuggingFace）：
  export HF_ENDPOINT=https://hf-mirror.com
"""

import os
import sys

# ── 确保 stdout 使用 UTF-8（解决 Windows GBK 编码问题） ────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 国内 HuggingFace 镜像 ────────────────────────────────
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def load_tokenizer() -> AutoTokenizer:
    """加载 Phi-3-mini-4k-instruct 的分词器。"""
    print(f"加载分词器：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  类型：{type(tokenizer).__name__}")
    print(f"  词汇表大小：{tokenizer.vocab_size:,}")
    print()

    # 打印特殊 token 信息
    print("  特殊 Token：")
    for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
        tok = getattr(tokenizer, name)
        tid = getattr(tokenizer, f"{name}_id")
        print(f"    {name:15s} = {str(tok):20s} (ID: {tid})")
    print()
    return tokenizer


def show_tokenizer_info(tokenizer: AutoTokenizer) -> None:
    """展示分词器基本信息。"""
    print("=" * 70)
    print("【分词器信息】")
    print("=" * 70)
    print(f"  模型：Phi-3-mini-4k-instruct")
    print(f"  算法：BPE (Byte-Pair Encoding)")
    print(f"  词汇表：{tokenizer.vocab_size:,} 个子词单元")
    print(f"  特殊 ID 范围：0 (unk) ~ {tokenizer.vocab_size} (eos)")
    print(f"  支持语言：中、英、代码等多语言（UTF-8 byte-level）")
    print()


def tokenize_and_show(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    show_ids: bool = True,
    show_decode: bool = True,
) -> dict:
    """
    分词并展示详细结果。

    返回：
      {"text": str, "tokens": list[str], "ids": list[int], "num_tokens": int}
    """
    BORDER = "=" * 70
    print(BORDER)
    print(f"  输入：{text}")
    print(f"  字符数：{len(text)}")
    print(BORDER)

    # ── 分词 ───────────────────────────────────────────────
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_with_special = tokenizer.encode(text)  # 带特殊 token

    # ── 打印 Token 序列 ────────────────────────────────────
    print(f"  分词结果（共 {len(tokens)} 个子词）：")
    header = f"  {'#':<4} {'子词':<16} {'ID':<8} {'bytes':<18} {'说明'}"
    print(header)
    print(f"  {'-' * 70}")
    for i, (token, tid) in enumerate(zip(tokens, ids)):
        # 显示 token 的字节表示（方便观察空格、特殊字符等）
        token_bytes = token.encode("utf-8")
        bytes_repr = " ".join(f"{b:02x}" for b in token_bytes)

        # 判断 token 类型
        if token.startswith("Ġ"):
            note = "空格前缀词"
        elif token.startswith("Ċ"):
            note = "换行符"
        elif len(token_bytes) >= 4 and all(b > 127 for b in token_bytes):
            note = "中文字符"
        elif all(c.isalpha() or c in ".-" for c in token):
            note = "英文子词"
        else:
            note = "标点/特殊"

        print(
            f"  {i:<4} {token:<16} {tid:<8} {bytes_repr:<18} {note}"
        )

    # ── 完整编码（含特殊 token） ────────────────────────────
    if show_ids:
        print()
        print(f"  含特殊 token 的 ID 序列（共 {len(ids_with_special)} 个）：")
        # 标记特殊 token
        labeled = []
        for tid in ids_with_special:
            if tid == tokenizer.bos_token_id:
                labeled.append(f"[BOS:{tid}]")
            elif tid == tokenizer.eos_token_id:
                labeled.append(f"[EOS:{tid}]")
            else:
                labeled.append(str(tid))
        print(f"    {' → '.join(labeled)}")
        print(f"    纯 ID 列表：{ids_with_special}")

    # ── 解码验证 ──────────────────────────────────────────
    if show_decode:
        decoded = tokenizer.decode(ids)
        decoded_with_special = tokenizer.decode(ids_with_special)
        print()
        print(f"  解码验证：")
        print(f"    无特殊 token 还原：「{decoded}」")
        print(f"    含特殊 token 还原：「{decoded_with_special}」")
        match = "完全匹配" if decoded == text else "有差异"
        print(f"    结果：{match}")
        if decoded != text:
            print(f"    差异说明：BPE 子词还原后可能因空格规范化有细微差异")

    print()
    return {"text": text, "tokens": tokens, "ids": ids, "num_tokens": len(ids)}


def demo_bpe_principle(tokenizer: AutoTokenizer) -> None:
    """展示 BPE 如何学习词合并。"""
    print("=" * 70)
    print("【BPE 原理演示】子词合并过程推演")
    print("=" * 70)
    text = "unbelievable"
    print(f"\n  目标词：{text}")
    print(f"  BPE 会将其拆为已知子词：{tokenizer.tokenize(text)}")
    print(f"  → IDs: {tokenizer.encode(text, add_special_tokens=False)}")
    print()

    text2 = "tokenization"
    print(f"  目标词：{text2}")
    print(f"  拆分：{tokenizer.tokenize(text2)}")
    print(f"  → IDs: {tokenizer.encode(text2, add_special_tokens=False)}")
    print()

    # 展示 tokenizer 如何处理罕见词
    text3 = "[HuggingFace]"
    print(f"  罕见词（Emoji+专名）：{text3}")
    t3 = tokenizer.tokenize(text3)
    i3 = tokenizer.encode(text3, add_special_tokens=False)
    print(f"  拆分：{t3}")
    print(f"  → IDs: {i3}")
    print(f"  解码还原：「{tokenizer.decode(i3)}」")
    print()


def demo_subword_merging(tokenizer: AutoTokenizer) -> None:
    """对比不同语言/频率词的子词切分粒度，直观展示 BPE 合并策略。"""
    print("=" * 70)
    print("【子词合并粒度对比】高频词 vs 低频词")
    print("=" * 70)

    # 对比高频英文词（可能是一个完整 token）vs 低频/罕见词（会被拆成多个子词）
    examples = [
        "the", "The", "deep", "learning", "deeplearning",
        "transformers", "tokenization", "tokenizers",
        "antidisestablishment", "pneumonoultramicroscopicsilicovolcanoconiosis",
    ]
    print(f"  {'词':<45} {'子词数':<8} {'子词切分':<30}")
    print(f"  {'-' * 85}")
    for word in examples:
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.encode(word, add_special_tokens=False)
        # 显示前几个 token 的简要形式
        token_summary = " | ".join(tokens[:5])
        if len(tokens) > 5:
            token_summary += f" ... (+{len(tokens)-5})"
        print(f"  {word:<45} {len(tokens):<8} {token_summary:<30}")

    print()
    print(f"  [说明] 日常高频词（如 the、deep）往往是单个 token，")
    print(f"     而罕见词或长词会被拆分为多个子词。")
    print(f"     这就是 BPE 的核心思想：平衡词表大小与 OOV 问题。")
    print()


def demo_decode(tokenizer: AutoTokenizer) -> None:
    """解码过程中的关键概念展示。"""
    print("=" * 70)
    print("【解码与 ID 转换】tokenizer.encode 和 tokenizer.decode")
    print("=" * 70)
    text = "我爱自然语言处理"
    ids = tokenizer.encode(text, add_special_tokens=False)

    print(f"  输入文本：{text}")
    print(f"  编码 IDs：{ids}")
    print()

    # 逐 token 解码
    print("  逐 token 解码（含特殊字节的显示为转义序列）：")
    for i, tid in enumerate(ids):
        t = tokenizer.decode([tid])
        try:
            t.encode("gbk")
        except UnicodeEncodeError:
            t = t.encode("unicode_escape").decode("ascii")
        print(f"    ID {tid:<6} -> [{t}]")
    print()

    # 重建
    reconstructed = tokenizer.decode(ids)
    print(f"  全部解码还原：「{reconstructed}」")
    print()


def demo_batch(tokenizer: AutoTokenizer) -> None:
    """批量编码与填充/截断。"""
    print("=" * 70)
    print("【批量编码】padding + truncation + return_tensors")
    print("=" * 70)
    texts = [
        "Hello world",
        "深度学习是人工智能的重要分支",
        "Phi-3-mini is a small language model by Microsoft.",
    ]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="np",
    )

    print(f"  batch size: {len(texts)}")
    print(f"  token IDs shape: {encoded['input_ids'].shape}")
    print(f"  attention_mask shape: {encoded['attention_mask'].shape}")
    print()

    for i, text in enumerate(texts):
        ids = encoded["input_ids"][i]
        mask = encoded["attention_mask"][i]
        actual_len = int(mask.sum())

        # 标记 PAD
        shown = []
        for j, tid in enumerate(ids):
            if mask[j] == 0:
                shown.append(f"[PAD]")
            elif tid == tokenizer.bos_token_id:
                shown.append(f"[BOS]")
            elif tid == tokenizer.eos_token_id:
                shown.append(f"[EOS]")
            else:
                shown.append(str(int(tid)))
        print(f"  [{i}] {text}")
        print(f"      IDs ({actual_len} tokens + padding): {shown}")
        print()


def main():
    tokenizer = load_tokenizer()
    show_tokenizer_info(tokenizer)

    # ── 示例文本（中/英/混） ────────────────────────────────
    texts = [
        "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>",
        # "我来到北京清华大学",
        # "深度学习是人工智能的重要分支",
        # "Hello World! 你好世界",
        # "The quick brown fox jumps over the lazy dog.",
        # "李华：人工智能是未来！\n王明：是的，我也这么认为。",
    ]

    # ── 逐一展示 ──────────────────────────────────────────
    print("=" * 70)
    print("【中英文分词展示】")
    print("=" * 70)
    for text in texts:
        tokenize_and_show(tokenizer, text)

    # ── BPE 原理演示 ──────────────────────────────────────
    demo_bpe_principle(tokenizer)

    # ── 高低频词对比 ──────────────────────────────────────
    demo_subword_merging(tokenizer)

    # ── 解码原理 ──────────────────────────────────────────
    demo_decode(tokenizer)

    # ── 批量编码 ──────────────────────────────────────────
    demo_batch(tokenizer)

    # ── 总结 ──────────────────────────────────────────────
    print("=" * 70)
    print("【总结】Phi-3-mini 分词器 vs jieba 分词器")
    print("=" * 70)
    comparisons = [
        ("算法", "BPE（Byte-Pair Encoding）", "基于词典的最大正向匹配"),
        ("词汇表", "~32k 子词单元", "~349k 完整词条"),
        ("OOV 问题", "[无] 罕见词可拆子词", "[有] 词典外词标为 OOV"),
        ("中文字粒度", "UTF-8 byte → 逐个 char", "基于词典的完整词"),
        ("英文", "子词级（如 playing→play+ing）", "仅支持完整词"),
        ("预处理", "空格前缀 Ġ、标点自动处理", "无特殊预处理"),
        ("编码输出", "tokenize() + encode()", "lcut() + 自行查表"),
        ("解码还原", "decode(ids) → 原文", "无内置解码"),
    ]
    print(f"  {'维度':<14} {'Phi-3 Tokenizer (BPE)':<32} {'Jieba 分词器':<25}")
    print(f"  {'-' * 75}")
    for dim, phi3, jieba_ in comparisons:
        print(f"  {dim:<14} {phi3:<32} {jieba_:<25}")
    print()
    print("  [核心区别]：")
    print("     - jieba：词典 + 规则驱动的词级分词")
    print("     - Phi-3：数据驱动的子词级分词（BPE）")
    print("     - BPE 的优势在于没有 OOV，且能处理任意语言")


if __name__ == "__main__":
    main()
