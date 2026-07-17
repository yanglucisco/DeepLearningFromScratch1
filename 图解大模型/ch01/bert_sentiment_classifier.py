"""
bert-base-multilingual-cased 舆情检测系统

用 BERT 嵌入 + 逻辑回归对中文文本进行情感二分类（正面/负面）。
使用 ChnSentiCorp 数据集（携程酒店评论）训练分类器。

流程：
  1. 加载预训练 BERT 模型（冻结权重）
  2. 批量提取训练/验证/测试文本的 [CLS] 嵌入向量
  3. 用 sklearn LogisticRegression 训练分类器
  4. 评估准确率、F1、混淆矩阵
  5. 对自定义舆情文本做预测演示

模型和数据均从 hf-mirror.com 下载。
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.ipc as ipc
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ── 常量 ─────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-multilingual-cased"
DATASET_REPO = "seamew/ChnSentiCorp"
BATCH_SIZE = 32           # CPU 批大小
MAX_LEN = 128             # 最大序列长度
RANDOM_STATE = 42


# ═════════════════════════════════════════════════════════════════
# 1. 模型加载
# ═════════════════════════════════════════════════════════════════

def load_model():
    """加载 bert-base-multilingual-cased（仅嵌入提取，冻结权重）。"""
    print("=" * 70)
    print("  bert-base-multilingual-cased 舆情检测系统")
    print("  基于 [CLS] 嵌入向量 + 逻辑回归的情感分类")
    print("=" * 70)
    print()

    print("[1/3] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  OK | vocab_size: {tokenizer.vocab_size:,}")

    print("[2/3] 加载模型...")
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK | 参数量：{total_params/1e6:.0f}M | 耗时：{time.time()-t0:.1f}秒")
    print()

    return model, tokenizer


# ═════════════════════════════════════════════════════════════════
# 2. 数据集加载（不使用 datasets 库，直接下载 .arrow 文件）
# ═════════════════════════════════════════════════════════════════

def _load_arrow_split(split: str) -> pd.DataFrame:
    """从 HuggingFace Hub 下载并读取一个 split 的 .arrow 文件。"""
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"chn_senti_corp-{split}.arrow",
        repo_type="dataset",
    )
    with open(path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    return table.to_pandas()


def load_dataset():
    """加载 ChnSentiCorp 数据集（训练/验证/测试）。"""
    print("[3/3] 加载 ChnSentiCorp 数据集（携程酒店评论）...")
    t0 = time.time()

    # 下载并读取 Arrow 文件
    df_train = _load_arrow_split("train")
    df_val = _load_arrow_split("validation")
    df_test = _load_arrow_split("test")

    print(f"  OK | 耗时：{time.time()-t0:.1f}秒")
    print(f"  OK | 训练集：{len(df_train):,} 条")
    print(f"  OK | 验证集：{len(df_val):,} 条")
    print(f"  OK | 测试集：{len(df_test):,} 条")
    print()

    # 检查标签分布
    for name, df in [("训练集", df_train), ("验证集", df_val), ("测试集", df_test)]:
        pos = (df["label"] == 1).sum()
        neg = (df["label"] == 0).sum()
        print(f"  ▸ {name}：正面 {pos} ({pos/len(df)*100:.0f}%) / 负面 {neg} ({neg/len(df)*100:.0f}%)")
    print()

    # 展示样本
    print("  ── 数据集样本预览 ──")
    for name, df in [("训练集", df_train), ("测试集", df_test)]:
        print(f"  ▸ {name}：")
        for i in range(3):
            label = "正面 👍" if df.iloc[i]["label"] == 1 else "负面 👎"
            text = df.iloc[i]["text"][:50] + ("..." if len(df.iloc[i]["text"]) > 50 else "")
            print(f"    [{label}] {text}")
        print()

    return df_train, df_val, df_test


# ═════════════════════════════════════════════════════════════════
# 3. 批量嵌入提取
# ═════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, tokenizer, texts: list[str],
                       desc: str = "") -> np.ndarray:
    """
    批量提取文本的 [CLS] 嵌入向量。

    BERT 的 [CLS] token 被设计为聚合整句信息的表示，
    可直接作为句子级特征使用。

    返回：
      shape (N, 768) 的 numpy 数组，每行是 L2 归一化的 [CLS] 向量
    """
    n = len(texts)
    all_embs = []
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  {desc} 嵌入提取（共 {n:,} 条，{n_batches} 批）...")
    t0 = time.time()

    for batch_idx, i in enumerate(range(0, n, BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]

        # 批量编码
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        )

        # 前向传播
        outputs = model(**inputs)

        # 取 [CLS] 位置的输出 → L2 归一化
        cls_vecs = F.normalize(outputs.last_hidden_state[:, 0, :], dim=1)
        all_embs.append(cls_vecs.cpu().numpy())

        # 进度显示（每 10% 或在最后一批时）
        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
            done = min(i + BATCH_SIZE, n)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remain = (n - done) / rate if rate > 0 else 0
            print(f"    ├ {done}/{n} ({done/n*100:.0f}%) | "
                  f"{rate:.0f} 条/秒 | 已用 {elapsed:.0f}s | 预计剩余 {remain:.0f}s")

    embs = np.vstack(all_embs)
    elapsed = time.time() - t0
    print(f"    └ ✓ {embs.shape[0]:,} × {embs.shape[1]} | "
          f"耗时 {elapsed:.0f}s | 平均 {embs.shape[0]/elapsed:.0f} 条/秒")
    print()

    return embs


# ═════════════════════════════════════════════════════════════════
# 4. 训练与评估
# ═════════════════════════════════════════════════════════════════

def train_and_evaluate(train_embs, train_labels,
                       val_embs, val_labels,
                       test_embs, test_labels,
                       test_texts):
    """训练逻辑回归并在测试集上评估。"""
    print("=" * 70)
    print("  [训练] 逻辑回归分类器")
    print("=" * 70)
    print()
    print(f"  训练集：{train_embs.shape[0]:,} 条 × {train_embs.shape[1]} 维")
    print(f"  验证集：{val_embs.shape[0]:,} 条")
    print(f"  测试集：{test_embs.shape[0]:,} 条")
    print()

    # ── 训练 ──
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(train_embs, train_labels)
    print(f"  ✓ 训练完成 | 耗时：{time.time()-t0:.3f}秒")
    print()

    # ── 评估 ──
    for name, embs, labels in [("验证集", val_embs, val_labels),
                                ("测试集", test_embs, test_labels)]:
        pred = clf.predict(embs)
        acc = accuracy_score(labels, pred)
        print(f"  ▸ {name}准确率：{acc:.4f} ({acc*100:.2f}%)")

    print()
    test_pred = clf.predict(test_embs)

    # ── 分类报告 ──
    print("  ── 测试集分类报告 ──")
    report = classification_report(
        test_labels, test_pred, target_names=["负面", "正面"], digits=4,
    )
    for line in report.split("\n"):
        if line.strip():
            print(f"  {line}")
    print()

    # ── 混淆矩阵 ──
    cm = confusion_matrix(test_labels, test_pred)
    tn, fp, fn, tp = cm.ravel()
    print("  ── 混淆矩阵 ──")
    print(f"              ┌─────────────┬─────────────┐")
    print(f"              │   预测负面   │   预测正面   │")
    print(f"  ────────────┼─────────────┼─────────────┤")
    print(f"   实际负面   │     {tn:>4}     │     {fp:>4}     │")
    print(f"  ────────────┼─────────────┼─────────────┤")
    print(f"   实际正面   │     {fn:>4}     │     {tp:>4}     │")
    print(f"  ────────────┴─────────────┴─────────────┘")
    print()
    print(f"  准确率：{(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")
    print(f"  精确率（正面）：{tp/(tp+fp)*100:.2f}%" if tp+fp > 0 else "")
    print(f"  召回率（正面）：{tp/(tp+fn)*100:.2f}%" if tp+fn > 0 else "")
    print()

    # ── 错误案例 ──
    print("  ── 分类错误案例（最多 8 条） ──")
    error_count = 0
    for i, (true_lbl, pred_lbl) in enumerate(zip(test_labels, test_pred)):
        if true_lbl != pred_lbl:
            true_str = "正面" if true_lbl == 1 else "负面"
            pred_str = "正面" if pred_lbl == 1 else "负面"
            text_short = test_texts[i][:55] + ("..." if len(test_texts[i]) > 55 else "")
            print(f"  ✗ 实际={true_str} / 预测={pred_str}")
            print(f"     \"{text_short}\"")
            error_count += 1
            if error_count >= 8:
                break
    if error_count == 0:
        print("  完美分类，无错误！🎉")
    print()

    return clf, test_pred


# ═════════════════════════════════════════════════════════════════
# 5. 舆情检测演示
# ═════════════════════════════════════════════════════════════════

def demo_predict(clf, model, tokenizer):
    """用训练好的分类器对自定义舆情文本做预测。"""
    print("=" * 70)
    print("  [舆情检测演示] 对自定义文本进行情感分析")
    print("=" * 70)
    print()

    test_texts = [
        # 正面
        "这家酒店服务态度非常好，房间干净整洁，强烈推荐",
        "产品质量超出预期，包装精美，物流也很快，好评",
        "今天天气真不错，心情很好",
        "食堂的饭菜越来越好了，价格也实惠",
        "这个 app 非常好用，界面简洁功能强大",
        "感谢客服耐心解答我的问题，服务态度一流",

        # 负面
        "服务太差了，等了两个小时还没人理，再也不来了",
        "这个产品质量有问题，用了三天就坏了，客服态度也很差",
        "交通太拥堵了，每天都迟到，受不了",
        "价格贵得离谱，完全不值这个价",
        "垃圾产品，强烈建议大家不要购买",
        "环境脏乱差，绝对不会再来第二次",

        # 中性/模糊
        "一般般吧，凑合能用",
        "东西收到了，还没试",
        "还可以，不算太好也不算太差",
        "这家店的位置不太好找，但东西还行",
        "开会讨论了一下，大家意见不太统一",
    ]

    print(f"  {'文本':<42} {'预测结果':<14} {'置信度':<8}")
    print(f"  {'-' * 66}")

    for text in test_texts:
        with torch.no_grad():
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=MAX_LEN,
            )
            outputs = model(**inputs)
            cls_vec = F.normalize(outputs.last_hidden_state[0, 0], dim=0)
            emb = cls_vec.cpu().numpy().reshape(1, -1)

        pred = clf.predict(emb)[0]
        proba = clf.predict_proba(emb)[0]
        confidence = max(proba)
        label_str = "👍 正面" if pred == 1 else "👎 负面"

        text_fmt = text[:40] + ("..." if len(text) > 40 else "")
        print(f"  {text_fmt:<42} {label_str:<14} {confidence:<8.1%}")

    print()
    print(f"  💡 模型能较好地区分正面和负面情感。")
    print(f"     中性和模糊表达通常置信度较低，说明模型不够确定。")
    print(f"     这是冻结 BERT + 逻辑回归的合理表现。")
    print()


# ═════════════════════════════════════════════════════════════════
# 6. 主函数
# ═════════════════════════════════════════════════════════════════

def main():
    # ── 加载模型和数据 ──
    model, tokenizer = load_model()
    df_train, df_val, df_test = load_dataset()

    # ── 提取嵌入向量 ──
    print("=" * 70)
    print("  [嵌入提取] 批量提取 [CLS] 向量")
    print("=" * 70)
    print()
    print(f"  使用 BERT [CLS] token 的输出作为句子表示（768 维）")
    print(f"  共需处理 {len(df_train) + len(df_val) + len(df_test):,} 条文本")
    print()

    train_embs = extract_embeddings(model, tokenizer, df_train["text"].tolist(), desc="训练集")
    val_embs = extract_embeddings(model, tokenizer, df_val["text"].tolist(), desc="验证集")
    test_embs = extract_embeddings(model, tokenizer, df_test["text"].tolist(), desc="测试集")

    # ── 训练与评估 ──
    clf, test_pred = train_and_evaluate(
        train_embs, df_train["label"].values,
        val_embs, df_val["label"].values,
        test_embs, df_test["label"].values,
        df_test["text"].tolist(),
    )

    # ── 舆情检测演示 ──
    demo_predict(clf, model, tokenizer)

    # ── 总结 ──
    test_acc = accuracy_score(df_test["label"].values, test_pred)
    print("=" * 70)
    print("  [总结]")
    print("=" * 70)
    print()
    print("  舆情检测系统工作流程：")
    print()
    print("  ① 加载预训练 BERT → 冻结权重")
    print(f"  ② 提取 ChnSentiCorp 训练集 [CLS] 嵌入（{len(train_embs):,}条 × {train_embs.shape[1]}维）")
    print("  ③ 训练 LogisticRegression 分类器")
    print(f"  ④ 测试集准确率：{test_acc*100:.2f}%")
    print("  ⑤ 对新文本：BERT 提取嵌入 → 分类器预测情感")
    print()
    print("  优势：")
    print("  ✓ 无需 GPU，CPU 即可快速训练和推理")
    print("  ✓ 可解释性强（逻辑回归权重对应每个 BERT 维度的贡献）")
    print("  ✓ 易于扩展（更换数据集、添加更多类别）")
    print()
    print("  局限与改进方向：")
    print("  ✗ 冻结 BERT 无法针对情感任务优化嵌入空间")
    print("  ✗ 对反讽、比喻等复杂表达可能不够准确")
    print("  ✗ 微调整个 BERT 通常能再提升 3-5 个百分点")
    print()
    print(f"  数据：hf-mirror.com  |  模型：{MODEL_NAME}  |  数据集：{DATASET_REPO}")
    print()


if __name__ == "__main__":
    main()
