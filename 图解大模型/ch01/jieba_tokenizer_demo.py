"""
Jieba 分词器示例 — 解析输入 → 生成词元 → 显示词元 ID

核心思路：读取 jieba 内置词典，用词典中的行号作为词元的 ID。
这样展示的才是"分词器内部的对应表"。
"""

import jieba
import os
import sys

# ── 确保 stdout 使用 UTF-8（解决 Windows GBK 编码问题） ────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# 1. 加载 jieba 内置词典，构建 {词: ID} 映射表
# ─────────────────────────────────────────────────────────────
DICT_PATH = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")


def load_jieba_vocab() -> dict[str, int]:
    """
    加载 jieba 内置词典，每一行格式为：词 词频 词性
    用行号（从 1 开始）作为该词的 ID。
    """
    vocab: dict[str, int] = {}
    with open(DICT_PATH, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            word = line.split()[0]          # 取第一个字段：词本身
            vocab[word] = line_no
    return vocab


def show_vocab_info(vocab: dict[str, int]) -> None:
    """打印 jieba 词典的统计信息。"""
    print("=" * 60)
    print("jieba 内置词典信息（行号 = 词元 ID）")
    print("=" * 60)
    print(f"  词典路径：{DICT_PATH}")
    print(f"  总词条数：{len(vocab):,}")
    print()

    # 按 ID 排序显示前 20 条
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    print(f"  前 20 条词条：")
    print(f"  {'ID':<8} {'词元':<12}")
    print(f"  {'-' * 22}")
    for word, wid in sorted_items[:20]:
        print(f"  {wid:<8} {word:<12}")
    print(f"  ...（共 {len(vocab):,} 条）")
    print()


# ─────────────────────────────────────────────────────────────
# 2. 分词 + ID 映射 + 展示
# ─────────────────────────────────────────────────────────────

def tokenize_and_show(
    text: str,
    vocab: dict[str, int],
    *,
    mode: str = "exact",
    show_pos: bool = False,
) -> list[int]:
    """
    分词 → 查 jieba 内置词典 ID → 展示结果。
    mode: 'exact' | 'full' | 'search'
    """
    border = "=" * 64
    print(border)
    print(f"  输入：{text}")
    mode_names = {"exact": "精确模式", "full": "全模式", "search": "搜索引擎模式"}
    print(f"  模式：{mode_names.get(mode, mode)}")
    print(border)

    # 分词
    if mode == "exact":
        tokens = jieba.lcut(text)
    elif mode == "full":
        tokens = jieba.lcut(text, cut_all=True)
    elif mode == "search":
        tokens = jieba.lcut_for_search(text)
    else:
        raise ValueError(f"未知模式：{mode}")

    # 查 jieba 内部词典的 ID，查不到则标记为 -1（词典外词）
    token_ids: list[int] = []
    for t in tokens:
        wid = vocab.get(t, -1)
        token_ids.append(wid)

    # 打印表格
    print(f"  {'#':<4} {'词元':<12} {'内部词典ID':<12} {'长度':<6} {'来源':<10}")
    print(f"  {'-' * 48}")
    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        source = "[词典内]" if tid != -1 else "[词典外]"
        tid_str = str(tid) if tid != -1 else " - "
        print(f"  {i:<4} {token:<12} {tid_str:<12} {len(token):<6} {source:<10}")
    print()

    # 统计覆盖率
    in_dict = sum(1 for tid in token_ids if tid != -1)
    total = len(token_ids)
    print(f"  [统计] 词典覆盖率：{in_dict}/{total}（{in_dict/total*100:.1f}%）")
    print()

    # 可选：词性标注
    if show_pos and mode == "exact":
        import jieba.posseg as pseg
        print(f"  --- 词性标注 ---")
        for w, flag in pseg.cut(text):
            print(f"    {w:<8} -> {flag}")
        print()

    return token_ids


# ─────────────────────────────────────────────────────────────
# 3. 主流程
# ─────────────────────────────────────────────────────────────

def main():
    # 加载 jieba 内部词典
    vocab = load_jieba_vocab()
    show_vocab_info(vocab)

    # ── 测试例句 ───────────────────────────────────────────
    texts = [
        "我来到北京清华大学",
        "深度学习是人工智能的重要分支",
        "我爱自然语言处理",
        "小明硕士毕业于中国科学院计算所后在日本京都大学深造",
        "江大桥是南京市著名旅游景点",
    ]

    # ── 精确模式分词 ───────────────────────────────────────
    print("=" * 60)
    print("【精确模式】分词 + jieba 内置词典 ID")
    print("=" * 60)
    for text in texts:
        tokenize_and_show(text, vocab, mode="exact")

    # ── 三种模式对比 ───────────────────────────────────────
    print("=" * 60)
    print("【模式对比】三种分词模式下的词元 ID")
    print("=" * 60)
    sample = "江大桥是南京市著名旅游景点"
    for mode in ("exact", "full", "search"):
        tokenize_and_show(sample, vocab, mode=mode)

    # ── 自定义词典影响 ─────────────────────────────────────
    print("=" * 60)
    print("【自定义词典】添加后 '江大桥' 的 ID 变化")
    print("=" * 60)

    # 自定义词典让 '江大桥' 整体成词
    import tempfile
    dic_path = os.path.join(tempfile.gettempdir(), "my_dict.txt")
    with open(dic_path, "w", encoding="utf-8") as f:
        f.write("江大桥 3 nr\n")

    # 先看自定义词加入前
    print("=> 加入前（'江大桥' 被拆为 '江' 和 '大桥'）：")
    tokenize_and_show("江大桥是南京市著名旅游景点", vocab, mode="exact")

    # 加载自定义词典
    jieba.load_userdict(dic_path)
    # 重新加载词典，把自定义词也纳入 vocab
    # 注：实际应用中自定义词的 ID 可以追加到现有 vocab 中
    custom_word = "江大桥"
    if custom_word not in vocab:
        vocab[custom_word] = len(vocab) + 1   # 追加一个新 ID

    print("=> 加入后（'江大桥' 被识别为整体，ID 追加到末尾）：")
    tokenize_and_show("江大桥是南京市著名旅游景点", vocab, mode="exact")

    # ── 批量输出 ───────────────────────────────────────────
    print("=" * 60)
    print("【批处理结果】JSON 格式输出")
    print("=" * 60)
    for text in texts:
        ids = tokenize_and_show(text, vocab, mode="exact")
        tokens = jieba.lcut(text)
        # 标记 OOV
        badge = [tid if tid != -1 else "OOV" for tid in ids]
        print(f"  {text}")
        print(f"    词元: {tokens}")
        print(f"    IDs:  {badge}")
        print()

    # ── 总结 ──────────────────────────────────────────────
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"  jieba 内置词典大小：{len(vocab):,} 条词")
    print(f"  ID 范围：1 ~ {len(vocab):,}")
    print(f"  OOV 标记：-1（该词在 jieba 内置词典中不存在）")
    print()
    print(f"  注意：jieba 本身只负责分词，不维护\"词 -> ID\"映射。")
    print(f"  本示例通过读取 jieba 内置词典的\"行号\"模拟 ID，")
    print(f"  方便展示分词结果的数值化。真实 NLP 流程中，一般")
    print(f"  会在分词后自行构建词表或使用预训练模型的 tokenizer。")


if __name__ == "__main__":
    main()
