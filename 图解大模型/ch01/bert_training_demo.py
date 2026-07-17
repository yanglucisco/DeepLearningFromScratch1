"""
bert-base-multilingual-cased 微调训练演示

演示如何用标注数据对预训练 BERT 进行微调（Fine-tuning）。
与 bert_movie_review_classifier.py 的零样本分类不同，
本示例实际训练模型的分类头，使模型适应特定任务。

训练流程：
  1. 准备中英文电影评价数据集（正面/负面）
  2. 用 AutoModelForSequenceClassification 加载 BERT + 分类头
  3. 分词 → 构建 DataLoader
  4. 训练循环：前向传播 → 计算 Loss → 反向传播 → 更新参数
  5. 评估：在测试集上计算准确率
  6. 保存模型 + 加载推理

核心概念：
  - 微调 ≠ 从头训练：BERT 的预训练权重作为初始化，只做少量 epoch
  - 分类头：在 BERT 的 [CLS] 输出上加一个全连接层 (768 → 2)
  - 学习率：微调通常用较小的学习率 (2e-5 ~ 5e-5)
  - 冻结策略：可选择性冻结底层，只训练顶层 + 分类头

模型和数据均从 hf-mirror.com 下载。
"""

import os
import sys
import time
import math
import copy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "bert-base-multilingual-cased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LENGTH = 128
WARMUP_STEPS = 0

# ── 数据集配置 ──────────────────────────────────────────────────────
# True  = 使用 IMDB 大型数据集（50K 条，效果更好但训练更慢）
# False = 使用手写的小数据集（32 条，快速验证）
USE_IMDB_DATASET = False

# 使用 IMDB 时随机抽取的样本数（设为 0 则使用全部 25K 条）
# CPU 建议 1000~2000，GPU 可用全部
IMDB_SAMPLE_SIZE = 2000

# 微调模型的本地保存路径
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "bert_finetuned_movie_review")

# 首次运行开关：设为 True 强制重新训练（即使已有保存的模型）
FORCE_RETRAIN = False


# ══════════════════════════════════════════════════════════════════════
#  1. 训练数据 — 中英文电影评价
# ══════════════════════════════════════════════════════════════════════

# 训练集：电影评价 + 情感标签（0=负面, 1=正面）
TRAIN_DATA = [
    # ── 正面评价 (label=1) ──
    ("This movie is absolutely fantastic, the best I have ever seen!", 1),
    ("演技精湛，剧情引人入胜，非常推荐", 1),
    ("Brilliant cinematography and a touching story. A masterpiece!", 1),
    ("导演的手法太棒了，每个镜头都充满艺术感", 1),
    ("An inspiring film with incredible performances from the cast.", 1),
    ("画面精美，配乐动听，今年最好的电影", 1),
    ("I loved every minute of this film, truly outstanding work.", 1),
    ("故事感人至深，演员演技在线，五星推荐", 1),
    ("A beautiful and moving piece of cinema. Highly recommended.", 1),
    ("节奏紧凑，剧情反转出人意料，年度最佳", 1),
    ("The acting was superb and the direction flawless.", 1),
    ("笑点密集又不失深度，难得一见的佳作", 1),
    ("This film restored my faith in modern cinema.", 1),
    ("情感真挚，细节丰富，值得反复品味", 1),
    ("A masterclass in storytelling. Every scene matters.", 1),
    ("配乐和画面的完美结合，视听盛宴", 1),

    # ── 负面评价 (label=0) ──
    ("Terrible movie, wasted two hours of my life.", 0),
    ("剧情无聊透顶，演员演技尴尬", 0),
    ("The plot made no sense and the dialogue was cringeworthy.", 0),
    ("特效粗糙，故事也毫无新意，浪费时间", 0),
    ("Disjointed storytelling and wooden acting. Avoid at all costs.", 0),
    ("剪辑混乱，完全看不懂在讲什么", 0),
    ("One of the worst films I have ever seen. Just awful.", 0),
    ("毫无逻辑的剧本，浪费了这么好的演员阵容", 0),
    ("Boring from start to finish. I fell asleep in the theater.", 0),
    ("烂片无疑，建议导演转行", 0),
    ("The special effects were laughable and the script was worse.", 0),
    ("全程尿点，毫无亮点可言", 0),
    ("A complete waste of talent and budget. Shameful.", 0),
    ("导演完全不知道自己在拍什么", 0),
    ("Pretentious and hollow. Style over substance at its worst.", 0),
    ("编剧应该被开除，这种剧本也敢拿出来拍", 0),
]

# 测试集：用于评估模型泛化能力（训练时不可见）
TEST_DATA = [
    ("An absolutely wonderful film that touched my heart deeply.", 1),
    ("这个电影太难看了，我看了半小时就睡着了", 0),
    ("A decent movie with some nice moments, but nothing special.", 1),
    ("The worst film I have seen this year. Complete disaster.", 0),
    ("演技在线，故事也很感人，值得一看", 1),
    ("特效不错但剧情太弱，总体还行", 1),
    ("Brilliant from start to finish. A true cinematic achievement.", 1),
    ("浪费时间金钱，电影院的椅子都比这片好看", 0),
    ("Not my cup of tea, but the acting was solid.", 1),
    ("这片子唯一的优点就是片尾字幕终于出来了", 0),
]


# ══════════════════════════════════════════════════════════════════════
#  1.5 加载 IMDB 数据集（从 hf-mirror.com 自动下载）
# ══════════════════════════════════════════════════════════════════════

def load_imdb_data(sample_size=2000, seed=42):
    """从 HuggingFace Hub 加载 IMDB 电影评价数据集。

    IMDB 数据集概况：
      - 50,000 条电影评价（训练 25K / 测试 25K）
      - 标签：0=负面 (评分≤4), 1=正面 (评分≥7)
      - 中立评价（评分 5~6）被过滤，确保标签明确
      - 数据来源：imdb.com 用户影评

    参数：
      sample_size: 随机抽取的样本数（减少 CPU 训练时间，0=全部使用）
      seed: 随机种子，保证每次抽取相同样本

    返回：
      train_data: list of (text, label)，加载失败返回 None
      test_data:  list of (text, label)，加载失败返回 None
    """
    print("=" * 70)
    print("  📦 加载 IMDB 电影评价数据集")
    print("=" * 70)
    print()
    print("  来源：huggingface.co/datasets/imdb（清华大学镜像加速）")
    print("  原始规模：训练 25,000 / 测试 25,000")
    print()

    # ── 检查依赖 ──
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ❌ 未安装 datasets 库")
        print("     请运行：pip install datasets")
        print("     回退到手写小数据集...")
        return None, None

    # ── 尝试加载（捕获所有异常，确保不回退失败） ──
    dataset = None
    errors = []

    # 方式 1：走 hf-mirror.com 镜像
    try:
        print("  [1/2] 从 hf-mirror.com 下载 IMDB 数据集...")
        # datasets 库也认 HF_ENDPOINT 环境变量
        dataset = load_dataset("stanfordnlp/imdb", trust_remote_code=True)
    except Exception as e:
        errors.append(f"镜像失败: {e}")

    # 方式 2：不走镜像，直连 HuggingFace（如果开了代理）
    if dataset is None:
        try:
            print("  ⚠ 镜像失败，尝试直连 HuggingFace...")
            old_endpoint = os.environ.pop("HF_ENDPOINT", None)
            dataset = load_dataset("stanfordnlp/imdb", trust_remote_code=True)
            # 恢复环境变量
            if old_endpoint:
                os.environ["HF_ENDPOINT"] = old_endpoint
        except Exception as e:
            errors.append(f"直连失败: {e}")
            if old_endpoint:
                os.environ["HF_ENDPOINT"] = old_endpoint

    # 方式 3：从本地缓存加载（如果之前成功下载过）
    if dataset is None:
        try:
            print("  ⚠ 尝试从本地缓存加载...")
            from datasets import load_from_disk
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/imdb")
            dataset = load_from_disk(cache_dir)
        except Exception:
            errors.append("本地缓存也不可用")

    # ── 全部失败 ──
    if dataset is None:
        print()
        print("  ❌ 加载 IMDB 数据集失败！")
        for err in errors:
            print(f"     • {err}")
        print()
        print("  🔧 排查建议：")
        print("     1. pip install datasets  (确保已安装)")
        print("     2. 检查网络连接（hf-mirror.com 是否可访问）")
        print("     3. 如用代理，检查代理设置")
        print("     4. 设 USE_IMDB_DATASET = False 使用手写数据")
        print()
        print("     ⬇ 自动回退到手写小数据集，继续运行...")
        print()
        return None, None

    # ── 加载成功 ──
    print(f"  OK | 训练集: {len(dataset['train']):,} 条")
    print(f"  OK | 测试集: {len(dataset['test']):,} 条")
    print()

    # ── 抽样（如果指定了 sample_size） ──
    if sample_size and sample_size > 0:
        print(f"  [2/2] 随机抽取 {sample_size:,} 条（加速 CPU 训练）...")
        half = sample_size // 2

        train_df = dataset["train"].to_pandas()
        pos_train = train_df[train_df["label"] == 1].sample(half, random_state=seed)
        neg_train = train_df[train_df["label"] == 0].sample(half, random_state=seed)
        train_sampled = list(pos_train.itertuples()) + list(neg_train.itertuples())

        test_half = min(500, sample_size // 4)
        test_df = dataset["test"].to_pandas()
        pos_test = test_df[test_df["label"] == 1].sample(test_half, random_state=seed)
        neg_test = test_df[test_df["label"] == 0].sample(test_half, random_state=seed)
        test_sampled = list(pos_test.itertuples()) + list(neg_test.itertuples())

        import random
        train_data = [(row.text, row.label) for row in train_sampled]
        test_data = [(row.text, row.label) for row in test_sampled]
        random.Random(seed).shuffle(train_data)
    else:
        print(f"  [2/2] 使用全部数据（训练较慢，效果更好）...")
        train_data = [(row["text"], row["label"]) for row in dataset["train"]]
        test_data = [(row["text"], row["label"]) for row in dataset["test"]]

    pos_count = sum(1 for _, lbl in train_data if lbl == 1)
    neg_count = sum(1 for _, lbl in train_data if lbl == 0)
    print(f"  OK | 训练集: {len(train_data):,} 条（正面 {pos_count} / 负面 {neg_count}）")
    print(f"  OK | 测试集: {len(test_data):,} 条")
    print()
    print(f"  💡 IMDB 数据集每条评价平均 200+ 单词，")
    print(f"     涵盖各种写作风格（专业影评人到普通观众），")
    print(f"     是情感分类任务的经典基准数据集。")
    print()

    return train_data, test_data


# ══════════════════════════════════════════════════════════════════════
#  2. 自定义 Dataset — 将文本转为模型输入
# ══════════════════════════════════════════════════════════════════════

class MovieReviewDataset(Dataset):
    """电影评价数据集。

    对每条评价：
      1. 用 BERT 分词器将文本转为 input_ids 和 attention_mask
      2. 截断/填充到固定长度 MAX_LENGTH
      3. 返回 {input_ids, attention_mask, labels}
    """

    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 分词：转 token IDs + attention mask
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════
#  3. 加载模型 — AutoModelForSequenceClassification
# ══════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """加载 BERT 分词器和带分类头的模型。

    加载优先级：
      1. 如果本地有微调好的模型 → 直接加载（无需联网）
      2. 否则从 HuggingFace 下载预训练模型（首次 ~700MB）

    AutoModelForSequenceClassification 在 BERT 的 [CLS] 输出上
    自动添加一个 dropout + 全连接层 (768 → 2) 用于二分类。
    """
    print("=" * 70)
    print("  bert-base-multilingual-cased 微调训练")
    print(f"  设备：{DEVICE}")
    print("=" * 70)
    print()

    # ── 检查是否有本地微调好的模型 ──
    if os.path.exists(SAVE_DIR) and not FORCE_RETRAIN:
        print("✅ 发现本地已微调的模型，跳过下载，直接加载！")
        print(f"   路径：{SAVE_DIR}")
        print()

        print("[1/2] 加载本地分词器...")
        tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
        print(f"  OK | vocab_size: {tokenizer.vocab_size:,}")
        print()

        print("[2/2] 加载本地微调模型...")
        model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  OK | 总参数量：{total_params / 1e6:.0f}M")
        print()
        print("  ⚡ 全程无需联网！模型已保存在本地。")
        print()

        model.to(DEVICE)
        return model, tokenizer

    # ── 否则从 HuggingFace 下载 ──
    print("[1/3] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  OK | vocab_size: {tokenizer.vocab_size:,}")
    print(f"  OK | pad_token: {tokenizer.pad_token} (ID:{tokenizer.pad_token_id})")
    print(f"  OK | cls_token: {tokenizer.cls_token} (ID:{tokenizer.cls_token_id})")
    print()

    print("[2/3] 加载 BERT + 分类头...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # 二分类：正面/负面
    )
    print(f"  OK | 分类头：768 → 2 (正面/负面)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK | 总参数量：{total_params / 1e6:.0f}M")
    print()

    # 展示分类头参数
    print("[3/3] 模型结构（分类头部分）：")
    print(f"  分类头：{model.classifier}")
    print()

    model.to(DEVICE)
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════
#  4. 训练辅助函数
# ══════════════════════════════════════════════════════════════════════

def calc_accuracy(logits, labels):
    """从 logits 计算准确率。"""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)


def evaluate(model, dataloader):
    """在给定 DataLoader 上评估模型。"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            correct, n = calc_accuracy(outputs.logits, labels)
            total_correct += correct
            total_samples += n

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    model.train()
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
#  5. 训练循环
# ══════════════════════════════════════════════════════════════════════

def train(model, tokenizer, train_data=None, test_data=None, epochs=None):
    """主训练流程。

    参数：
      train_data, test_data: 如果不传，默认使用全局的 TRAIN_DATA / TEST_DATA
      epochs: 如果不传，默认使用全局 EPOCHS

    步骤：
      1. 构建 DataLoader
      2. 设置优化器（AdamW）和学习率调度器
      3. 每个 epoch：训练 → 评估 → 打印指标
      4. 保存最佳模型
    """
    if train_data is None:
        train_data = TRAIN_DATA
    if test_data is None:
        test_data = TEST_DATA
    if epochs is None:
        epochs = EPOCHS

    print("=" * 70)
    print("  [开始训练]")
    print("=" * 70)
    print()
    print(f"  训练集大小：{len(train_data)} 条")
    print(f"  测试集大小：{len(test_data)} 条")
    print(f"  Batch Size：{BATCH_SIZE}")
    print(f"  学习率：{LEARNING_RATE}")
    print(f"  Epochs：{epochs}")
    print(f"  Max Length：{MAX_LENGTH}")
    print()

    # ── 构建 DataLoader ──
    train_dataset = MovieReviewDataset(train_data, tokenizer)
    test_dataset = MovieReviewDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练批次数：{len(train_loader)}")
    print(f"  测试批次数：{len(test_loader)}")
    print()

    # ── 优化器 + 学习率调度器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # ── 训练前评估（基线） ──
    print("  ── 训练前（随机分类头） ──")
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"  测试 Loss: {test_loss:.4f} | 准确率: {test_acc:.2%}")
    print(f"  (随机初始化时准确率约 50%，相当于瞎猜)")
    print()

    # ── 训练循环 ──
    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # 统计
            epoch_loss += loss.item()
            correct, n = calc_accuracy(outputs.logits, labels)
            epoch_correct += correct
            epoch_samples += n

        # ── Epoch 结束 ──
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_samples

        # 评估
        test_loss, test_acc = evaluate(model, test_loader)

        elapsed = time.time() - t0
        improved = "★" if test_acc > best_acc else " "
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"  Epoch {epoch + 1}/{epochs} | "
            f"耗时: {elapsed:.1f}s | "
            f"训练 Loss: {avg_train_loss:.4f} | "
            f"训练 Acc: {train_acc:.2%} | "
            f"测试 Loss: {test_loss:.4f} | "
            f"测试 Acc: {test_acc:.2%} {improved}"
        )

    print()
    print(f"  最佳测试准确率：{best_acc:.2%}")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print()
    print("  💡 训练完成！BERT 从预训练的通用语义理解，")
    print("     经过微调适应了电影评价情感分类任务。")
    print()

    # ── 保存模型到本地 ──
    print(f"  💾 保存微调模型到本地：{SAVE_DIR}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"  ✅ 保存完成！下次运行将直接加载本地模型，无需重新训练。")
    print(f"     ├─ {os.path.join(SAVE_DIR, 'config.json')}")
    print(f"     ├─ {os.path.join(SAVE_DIR, 'model.safetensors')}  (约 700MB)")
    print(f"     └─ {os.path.join(SAVE_DIR, 'tokenizer.json')}")
    print()

    return best_acc


# ══════════════════════════════════════════════════════════════════════
#  6. 推理演示 — 用训练好的模型预测
# ══════════════════════════════════════════════════════════════════════

def demo_inference(model, tokenizer):
    """用微调后的模型对新的电影评价做推理。"""
    print("=" * 70)
    print("  [推理演示] 微调后模型的预测")
    print("=" * 70)
    print()

    # 新评价（训练过程中未出现过的）
    new_reviews = [
        "An emotional rollercoaster with a powerful message. Loved it!",
        "这部电影的结局出乎意料，让人久久不能平静，力荐！",
        "The pacing was off and the ending felt rushed. Disappointing.",
        "典型的烂片，剧情老套，演员面瘫，不值票价",
        "While not perfect, the film has genuine heart and charm.",
        "可圈可点之处不多，但也不算太难看",
        "还行吧，一般般",
    ]

    model.eval()
    for review in new_reviews:
        inputs = tokenizer(
            review,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, 2)
            probs = F.softmax(logits, dim=-1)
            pred_label = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, pred_label].item()

        label_str = "🎬 正面" if pred_label == 1 else "👎 负面"
        print(f"  评价：{review}")
        print(f"  预测：{label_str}  (置信度 {confidence:.2%})")
        print(f"  概率：负面={probs[0,0]:.4f}  正面={probs[0,1]:.4f}")
        print()

    # 注意：训练集很小（16条），真实场景需要更多数据


# ══════════════════════════════════════════════════════════════════════
#  7. 对比实验：冻结底层 vs 全量微调
# ══════════════════════════════════════════════════════════════════════

def demo_frozen_layers(model, tokenizer, train_data=None, test_data=None):
    """演示冻结 BERT 底层、仅训练分类头的效果。

    这是一种常见的微调策略：
      - 冻结底层：只训练分类头（快速，但效果有限）
      - 全量微调：训练所有参数（效果好，需要更多资源）
    """
    if train_data is None:
        train_data = TRAIN_DATA
    if test_data is None:
        test_data = TEST_DATA

    print("=" * 70)
    print("  [对比实验] 冻结底层 vs 全量微调")
    print("=" * 70)
    print()
    print("  对比两种策略：")
    print("    a) 仅训练分类头（冻结 BERT 所有层）")
    print("    b) 全量微调（刚才的默认行为）")
    print()

    # ── 策略 A：冻结 BERT 底层 ──
    print("  ── 策略 A：仅训练分类头（冻结 BERT 底层） ──")
    print()

    frozen_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    frozen_model.to(DEVICE)

    # 冻结 BERT 的所有参数
    for name, param in frozen_model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in frozen_model.parameters())
    print(f"  可训练参数：{trainable_params:,} / {total_params:,} "
          f"({trainable_params / total_params * 100:.1f}%)")
    print(f"  冻结了 BERT 的 {total_params - trainable_params:,} 个参数")

    # 对于 IMDB 大数据集，只取少量做快速对比
    if len(train_data) > 500:
        import random
        train_data = random.Random(42).sample(train_data, 200)
        print(f"  (从大数据集抽样 200 条做快速对比)")
    if len(test_data) > 500:
        import random
        test_data = random.Random(42).sample(test_data, 100)

    train_dataset = MovieReviewDataset(train_data, tokenizer)
    test_dataset = MovieReviewDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, frozen_model.parameters()),
        lr=1e-3,  # 仅训练分类头时可用更大的学习率
    )

    num_epochs = min(3, EPOCHS)
    for epoch in range(num_epochs):
        frozen_model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = frozen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss, test_acc = evaluate(frozen_model, test_loader)
        print(f"  Epoch {epoch + 1}/{num_epochs} | 测试 Loss: {test_loss:.4f} | 测试 Acc: {test_acc:.2%}")

    print()
    print(f"  ── 对比总结 ──")
    print(f"  策略 A（仅分类头）：")
    print(f"    ✓ 训练极快（只更新 ~1.5K 参数）")
    print(f"    ✓ 不易过拟合小数据集")
    print(f"    ✗ 效果受限于预训练特征")
    print()
    print(f"  策略 B（全量微调）：")
    print(f"    ✓ 效果更好（BERT 适配任务）")
    print(f"    ✗ 训练更慢（更新 ~178M 参数）")
    print(f"    ✗ 小数据集可能过拟合")
    print()
    print(f"  💡 选择策略取决于：数据集大小、计算资源、任务难度。")
    print()


# ══════════════════════════════════════════════════════════════════════
#  8. 数据增强演示
# ══════════════════════════════════════════════════════════════════════

def demo_data_augmentation():
    """演示训练数据的关键概念。

    在 NLP 中，数据质量远比模型架构重要。展示一些关键实践：
    """
    print("=" * 70)
    print("  [训练数据最佳实践]")
    print("=" * 70)
    print()
    print("  ── 数据量与效果的关系 ──")
    print()
    print("  数据集大小    微调策略          预期效果")
    print("  " + "-" * 56)
    print("  < 100 条      仅训练分类头      可用于原型验证")
    print("  100~1K 条     全量微调 + 正则化  基本可用")
    print("  1K~10K 条     全量微调           效果较好")
    print("  > 10K 条      全量微调           生产级效果")
    print()
    print("  ── 数据质量清单 ──")
    print("  ☐ 类别平衡（正面/负面比例接近 1:1）")
    print("  ☐ 覆盖多种表达方式（强烈/温和/反讽）")
    print("  ☐ 多语言覆盖（手写数据中英混合，IMDB 纯英文）")
    print("  ☐ 测试集与训练集无重叠")
    print("  ☐ 测试集包含模型未见过的表达方式")
    print()
    print("  ── 两种数据源的对比 ──")
    print()
    pos_hand = sum(1 for _, label in TRAIN_DATA if label == 1)
    neg_hand = sum(1 for _, label in TRAIN_DATA if label == 0)
    print(f"  手写数据集：{len(TRAIN_DATA)} 条（正面 {pos_hand} / 负面 {neg_hand}）")
    print(f"    ✓ 中英混合 → 适合多语言场景")
    print(f"    ✗ 仅 32 条 → 容易过拟合，泛化能力弱")
    print(f"    ✗ 表达方式单一 → 对\"还行\"等中性表达束手无策")
    print()
    if USE_IMDB_DATASET:
        print(f"  IMDB 数据集：{IMDB_SAMPLE_SIZE:,} 条（本次抽样）")
    print(f"  IMDB 数据集原始：50,000 条（训练 25K / 测试 25K）")
    print(f"    ✓ 数据量大 → 泛化能力强，不易过拟合")
    print(f"    ✓ 表达多样 → 覆盖各种风格和强度")
    print(f"    ✓ 工业标准 → 学术论文常用基准")
    print(f"    ✗ 仅英文 → 中文场景需额外数据")
    print()
    print("  ── 如何获取更多训练数据 ──")
    print("  1. HuggingFace Hub (hf-mirror.com)")
    print("     • imdb — 50K 电影评价（本演示已集成）")
    print("     • rotten_tomatoes — 10K 烂番茄影评")
    print("     • amazon_reviews_multi — 多语言 Amazon 评价")
    print("     • yelp_review_full — 700K Yelp 评价")
    print("  2. 数据增强技术")
    print("     • 同义替换：用同义词替换关键词")
    print("     • 回译：中→英→中，英→中→英")
    print("     • Easy Data Augmentation (EDA)")
    print("  3. 主动学习")
    print("     • 先用少量数据训练 → 对大量未标注数据预测")
    print("     • 找出模型最不确定的样本 → 人工标注 → 加入训练")
    print()


# ══════════════════════════════════════════════════════════════════════
#  9. 演示：加载已保存的本地模型推理
# ══════════════════════════════════════════════════════════════════════

def demo_load_saved_model():
    """演示从本地加载已微调的模型，完全无需联网。"""
    print("=" * 70)
    print("  [演示] 从本地加载已保存的微调模型")
    print("=" * 70)
    print()
    print("  模拟第二次运行的场景：模型已在本地，无需下载。")
    print()

    if not os.path.exists(SAVE_DIR):
        print("  ⚠ 尚未保存微调模型，请先运行训练。")
        print()
        return

    print(f"  📂 从 {SAVE_DIR} 加载模型...")
    print()

    # 直接从本地目录加载（不访问网络）
    local_model = AutoModelForSequenceClassification.from_pretrained(
        SAVE_DIR,
        local_files_only=True,  # 关键参数：强制只使用本地文件
    )
    local_tokenizer = AutoTokenizer.from_pretrained(
        SAVE_DIR,
        local_files_only=True,
    )
    local_model.to(DEVICE)
    local_model.eval()

    print(f"  ✅ 加载成功！全程零网络访问。")
    print()

    # 用本地模型做几条推理
    test_reviews = [
        "这部电影拍得太好了，强烈推荐！",
        "What a boring and pointless film. I regret watching it.",
        "中规中矩，无功无过，可以一看",
    ]

    print("  推理测试：")
    for review in test_reviews:
        inputs = local_tokenizer(
            review, truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = local_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(outputs.logits, dim=-1).item()

        label_str = "🎬 正面" if pred == 1 else "👎 负面"
        print(f"    {label_str} | {review} "
              f"(负面:{probs[0,0]:.2f} 正面:{probs[0,1]:.2f})")

    print()
    print(f"  💡 关键点总结：")
    print(f"    1. 第一次运行：下载预训练模型 → 微调 → save_pretrained() 保存到本地")
    print(f"    2. 第二次运行：from_pretrained(local_path) 直接从本地加载")
    print(f"    3. 预训练模型缓存在 ~/.cache/huggingface/hub/（HuggingFace 自动管理）")
    print(f"    4. 微调模型保存在 {os.path.basename(SAVE_DIR)}（你指定的路径）")
    print(f"    5. 设置 local_files_only=True 可确保不走网络")
    print()


# ══════════════════════════════════════════════════════════════════════
#  10. 主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 70)
    print("   bert-base-multilingual-cased 微调训练演示")
    print("   从预训练模型到任务适配 — 电影评价情感分类")
    print("=" * 70)
    print()
    print("  核心问题：BERT 是通用语言模型，如何适配到具体任务？")
    print()
    print("  答案：微调（Fine-tuning）")
    print("    - 在 BERT 顶层加一个分类头（768 → 2）")
    print("    - 用标注数据同时训练分类头和微调 BERT 参数")
    print("    - 预训练权重提供语义基础，微调让模型适配任务")
    print()
    print("  ── 关于模型缓存与保存 ──")
    print("  📦 预训练模型：HuggingFace 自动缓存到 ~/.cache/huggingface/hub/")
    print("     首次下载 ~700MB，后续 from_pretrained() 直接用缓存")
    print(f"  💾 微调模型：训练完成后 save_pretrained() 保存到本地目录")
    print(f"     路径：{SAVE_DIR}")
    print(f"     下次运行时自动检测并跳过训练，直接加载")
    print()

    # ── 选择训练数据源 ──
    train_data = TRAIN_DATA
    test_data = TEST_DATA
    data_source_name = "手写小数据集（32 条）"

    if USE_IMDB_DATASET:
        imdb_train, imdb_test = load_imdb_data(sample_size=IMDB_SAMPLE_SIZE)
        if imdb_train is not None and imdb_test is not None:
            train_data = imdb_train
            test_data = imdb_test
            data_source_name = f"IMDB 数据集（{len(train_data):,} 条）"
        else:
            print("  ⚠ IMDB 加载失败，回退到手写小数据集")
            print()

    # ── 加载模型（优先本地微调模型 → 回退到 HuggingFace 下载） ──
    model, tokenizer = load_model_and_tokenizer()

    # ── 判断是否需要训练 ──
    if os.path.exists(SAVE_DIR) and not FORCE_RETRAIN:
        # 已有微调模型，跳过训练
        print("=" * 70)
        print("  ⏭ 跳过训练（本地已存在微调模型）")
        print(f"     如需重新训练，设置 FORCE_RETRAIN = True 或删除")
        print(f"     {os.path.basename(SAVE_DIR)} 目录")
        print("=" * 70)
    else:
        # 执行训练
        if USE_IMDB_DATASET:
            print(f"  🏷 数据源：{data_source_name}")
            print()

        _ = train(model, tokenizer, train_data=train_data, test_data=test_data)

    # ── 推理演示 ──
    demo_inference(model, tokenizer)

    # ── 本地加载演示 ──
    # demo_load_saved_model()

    # ── 对比实验 ──
    # demo_frozen_layers(model, tokenizer)

    # ── 数据最佳实践 ──
    # demo_data_augmentation()

    # ── 总结 ──
    print("=" * 70)
    print("  [总结] 微调 BERT 的完整流程")
    print("=" * 70)
    print()
    print("  1. 数据准备")
    if USE_IMDB_DATASET:
        print("     └─ IMDB 数据集（25K 条）→ load_dataset('imdb')")
    else:
        print("     └─ 手写标注数据（32 条）→ 分词 → DataLoader")
    print()
    print("  2. 模型准备")
    print("     └─ 加载预训练 BERT → 添加分类头 → 移至设备")
    print()
    print("  3. 训练配置")
    print("     └─ AdamW 优化器 + 线性衰减学习率 + 梯度裁剪")
    print()
    print("  4. 训练循环")
    print("     └─ for epoch: for batch: forward → loss → backward → step")
    print()
    print("  5. 保存模型到本地")
    print("     └─ model.save_pretrained() + tokenizer.save_pretrained()")
    print(f"     └─ 路径：{SAVE_DIR}")
    print()
    print("  6. 下次运行时加载本地模型")
    print("     └─ from_pretrained(local_path) — 无需联网，无需重新训练")
    print()
    print("  ── 关键超参数 ──")
    print(f"  学习率：{LEARNING_RATE}（微调用 2e-5~5e-5）")
    print(f"  Batch Size：{BATCH_SIZE}（取决于 GPU 内存）")
    print(f"  Epochs：{EPOCHS}（数据集越大，epochs 可越少）")
    print(f"  Max Length：{MAX_LENGTH}（根据任务调整）")
    if USE_IMDB_DATASET:
        print(f"  IMDB 抽样：{IMDB_SAMPLE_SIZE:,} 条（设 0 使用全部 25K）")
    print()
    print("  ── 微调 vs 零样本 vs 从头训练 ──")
    print("  零样本（本目录 bert_movie_review_classifier.py）：")
    print("    ✓ 无需训练数据 | ✗ 精度较低")
    print("  微调（本文件）：")
    print("    ✓ 精度高 | ✓ 适应各种任务 | ✗ 需要标注数据")
    print("  从头训练 BERT：")
    print("    ✗ 需海量数据和算力（不推荐）")
    print()
    print(f"  模型：{MODEL_NAME} (~178M 参数)")
    print(f"  设备：{DEVICE}")
    print()


if __name__ == "__main__":
    main()
