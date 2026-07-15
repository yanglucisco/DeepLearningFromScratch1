"""
Qwen2.5-1.5B-Instruct LLM 推理示例（CPU 友好版）
  1. 加载模型和分词器（~3GB，CPU 约 10~20 秒出结果）
  2. 输入文本 → 应用聊天模板 → tokenize → 显示 ID 序列
  3. model.generate() 自回归生成
  4. 逐步解码展示 + 最终输出

首次运行会自动下载约 3GB 模型文件（HuggingFace 国内镜像）。
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
USER_MSG = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."


def load_model_and_tokenizer():
    """加载 Qwen2.5-1.5B-Instruct（~1.5B 参数，CPU 约 10~20 秒）。"""
    print("=" * 70)
    print(f"  模型：{MODEL_NAME}")
    print(f"  PyTorch: {torch.__version__} (CPU)")
    print(f"  参数量：~1.5B（GPT-2 的 10 倍，但具备指令遵循能力）")
    print(f"  大小：约 3 GB（量化后约 1.8 GB）")
    print("=" * 70)
    print()

    print("[1/2] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  OK | vocab_size: {tokenizer.vocab_size:,}")
    print(f"  OK | pad_token: {tokenizer.pad_token} (ID:{tokenizer.pad_token_id})")
    print(f"  OK | eos_token: {tokenizer.eos_token} (ID:{tokenizer.eos_token_id})")
    print()

    print("[2/2] 加载模型（首次会下载约 3GB 模型文件）...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,  # half precision，节省内存
        low_cpu_mem_usage=True,
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK | 参数量：{total_params / 1e9:.2f}B")
    print(f"  OK | 耗时：{time.time() - t0:.1f} 秒")
    print()

    return model, tokenizer


def tokenize_input(tokenizer, user_msg: str) -> tuple[list[int], str]:
    """
    将用户消息按 Qwen 聊天模板格式化，分词并展示。
    返回 (ids, formatted_prompt)。
    """
    print("=" * 70)
    print("  [分词阶段]")
    print("=" * 70)
    print(f"  用户消息：{user_msg}\n")

    # 按聊天模板组装（自动添加 system/user/assistant 标记）
    messages = [{"role": "user", "content": user_msg}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(f"  模板化后的完整输入：")
    print(f"  {formatted}")
    print()

    tokens = tokenizer.tokenize(formatted)
    ids = tokenizer.encode(formatted)
    ids_no_special = tokenizer.encode(formatted, add_special_tokens=False)

    print(f"  字符数：{len(formatted)} | 子词数：{len(tokens)} | ID数：{len(ids)}")
    print()

    # 显示前 30 个 token 的拆解
    show_count = min(30, len(tokens))
    print(f"  Token 拆解（前 {show_count} 个）：")
    for i in range(show_count):
        tok = tokens[i]
        tid = ids_no_special[i]
        note = "词首/空格" if tok.startswith("Ġ") else "特殊标记" if tok.startswith("<|im") else "普通子词"
        print(f"    {i:<3} {tok:<20} ID:{tid:<8} ({note})")
    if len(tokens) > show_count:
        print(f"    ...（共 {len(tokens)} 个）")

    # 用标记标记特殊 token
    seq = []
    for tid in ids:
        if tid == tokenizer.bos_token_id:
            seq.append("[BOS]")
        elif tid == tokenizer.eos_token_id:
            seq.append("[EOS]")
        else:
            seq.append(str(tid))
    print(f"\n  完整 ID 序列（{len(seq)} 个）：")
    print(f"  {' -> '.join(seq)}")
    print()

    return ids, formatted


def generate_text(model, tokenizer, input_ids: list[int], formatted_prompt: str, max_new_tokens=120):
    """模型推理生成并展示逐步解码过程。"""
    print("=" * 70)
    print("  [生成阶段] 自回归逐 token 生成")
    print("  Qwen2.5-1.5B CPU 推理，约 10~20 秒")
    print("=" * 70)
    print()

    t0 = time.time()

    with torch.no_grad():
        output = model.generate(
            input_ids=torch.tensor([input_ids]),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )

    elapsed = time.time() - t0

    generated_ids = output[0][len(input_ids):].tolist()

    print(f"  生成完成！")
    print(f"  | 输入 tokens：{len(input_ids)}")
    print(f"  | 生成 tokens：{len(generated_ids)}")
    print(f"  | 耗时：{elapsed:.1f} 秒")
    print(f"  | 速度：{len(generated_ids) / elapsed:.1f} token/s")
    print()

    # ── 逐 token 解码展示 ─────────────────────────────────
    print("-" * 70)
    print("  逐步生成过程：")
    print("-" * 70)
    show_count = min(20, len(generated_ids))
    for i in range(show_count):
        tid = generated_ids[i]
        token_text = tokenizer.decode([tid])
        if tid == tokenizer.eos_token_id:
            label = "<EOS>"
        elif token_text == "\n":
            label = "\\n"
        elif token_text.startswith(" "):
            label = repr(token_text)
        else:
            label = token_text
        print(f"    Step {i+1:>3}: ID {tid:<6} -> {label}")
    if len(generated_ids) > show_count:
        print(f"    ...（剩余 {len(generated_ids) - show_count} 个 token）")
    print()

    # ── 完整输出 ──────────────────────────────────────────
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("=" * 70)
    print("  [最终输出] Qwen2.5 生成的邮件")
    print("=" * 70)
    print()
    print(full_text)
    print()

    return full_text


def main():
    model, tokenizer = load_model_and_tokenizer()

    ids, formatted = tokenize_input(tokenizer, USER_MSG)

    result = generate_text(model, tokenizer, ids, formatted)

    print("=" * 70)
    print("  [流程总结]")
    print("=" * 70)
    print("  1. 模型加载 | Qwen2.5-1.5B-Instruct（1.5B params）")
    print(f"  2. 聊天模板 | 自动添加 <|im_start|> 标记")
    print(f"  3. 分词编码 | {len(ids)} tokens -> ID 序列")
    print("  4. 模型推理 | 自回归生成 (temperature=0.7, top_k=50)")
    print("  5. 逐 token | 每步生成概率最高的子词")
    print("  6. 解码输出 | token IDs -> 可读文本")
    print()
    print(f"  💡 Qwen2.5-1.5B 是 Instruct 模型，能理解指令、生成符合要求的文本。")
    print(f"     相比 GPT-2 的随机续写，Instruct 模型的输出更有针对性。")


if __name__ == "__main__":
    main()
