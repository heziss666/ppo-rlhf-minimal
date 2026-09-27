"""评测 PPO-RLHF：SFT 策略（PPO 前）vs SFT+PPO 策略（PPO 后）。

任务指标（沿用 SFT 项目）：
  JSON Valid / Tool Accuracy / Arguments Accuracy / Exact Match
再加 Reward Model 打分：
  Average RM Reward

用法：
  python evaluate.py --base                # PPO 之前的 SFT 策略（基线）
  python evaluate.py --adapter ./output    # PPO 之后的策略
  python evaluate.py                       # 不传参数时自动检测 ./output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
BASE = "Qwen/Qwen3-0.6B"
SFT_ADAPTER = HERE.parent / "sft_tool_call" / "output"
RM_ADAPTER = HERE.parent / "reward-model-minimal" / "output"
DATA = HERE / "test.jsonl"
DEFAULT_ADAPTER = HERE / "output"

SYS = "你是一个工具调用助手。根据用户需求选择工具并提取参数，只输出合法JSON，不要解释。"
CHAT_KWARGS = {"enable_thinking": False}
MAX_NEW_TOKENS = 64

# 固定样例：PPO 前后各看一次，直观对比回答和 reward 的变化
DEMO = "帮我查一下杭州明天天气"


def messages(user):
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


def extract_json(text):
    """从模型输出里抽出第一个完整的 {...} 对象；失败返回 None。"""
    text = re.sub(r"<think>.*?</think>", "", text.replace("<|im_end|>", ""), flags=re.S)
    for m in re.finditer(r"\{", text):
        depth, in_str, esc = 0, False, False
        for i in range(m.start(), len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[m.start():i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def load_models(ppo_adapter):
    """策略 = Base + SFT adapter（merge） [+ PPO adapter]；Reward = Base seq-cls + RM adapter。"""
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, str(SFT_ADAPTER)).merge_and_unload()
    if ppo_adapter:
        policy = PeftModel.from_pretrained(policy, str(ppo_adapter))
    policy = policy.to("cuda").eval()

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    rm_base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, dtype=torch.bfloat16)
    reward = PeftModel.from_pretrained(rm_base, str(RM_ADAPTER))
    reward.config.pad_token_id = tok.pad_token_id
    reward = reward.to("cuda").eval()
    return tok, policy, reward


def generate(tok, policy, user):
    """确定性生成（do_sample=False），返回本次新生成的文本。"""
    text = tok.apply_chat_template(messages(user), tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(policy.device)
    with torch.no_grad():
        out = policy.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def reward_of(tok, reward, user, response):
    """RM 给 prompt + response 打分。"""
    text = tok.apply_chat_template(messages(user) + [{"role": "assistant", "content": response}], tokenize=False)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(reward.device)
    with torch.no_grad():
        return reward(**inputs).logits.squeeze(-1).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="只测 PPO 之前的 SFT 策略")
    ap.add_argument("--adapter", default=None, help="PPO LoRA adapter 目录")
    a = ap.parse_args()

    adapter = None if a.base else (a.adapter or (str(DEFAULT_ADAPTER) if DEFAULT_ADAPTER.exists() else None))
    label = f"SFT + PPO({adapter})" if adapter else "SFT（PPO 之前）"
    print(f"===== 评测对象：{label} =====")

    tok, policy, reward = load_models(adapter)
    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]

    stats = {"json_valid": 0, "tool": 0, "args": 0, "exact": 0}
    rewards, failures = [], []
    for i, row in enumerate(rows):
        target = json.loads(row["answer"])
        raw = generate(tok, policy, row["prompt"])
        pred = extract_json(raw)
        rewards.append(reward_of(tok, reward, row["prompt"], raw))
        if pred is not None:
            stats["json_valid"] += 1
            stats["tool"] += pred.get("tool") == target["tool"]
            stats["args"] += pred.get("arguments") == target["arguments"]
            stats["exact"] += pred == target
        if pred != target:
            failures.append((row["prompt"], target, raw, pred))

    n = len(rows)
    print(f"\nTotal: {n}")
    print(f"JSON Valid Rate    : {stats['json_valid'] / n:6.1%}")
    print(f"Tool Accuracy      : {stats['tool'] / n:6.1%}")
    print(f"Arguments Accuracy : {stats['args'] / n:6.1%}")
    print(f"Exact Match        : {stats['exact'] / n:6.1%}")
    print(f"Average RM Reward  : {sum(rewards) / n:6.2f}")
    print("（四项任务指标以全部样本为分母；RM Reward 只作参考，不能单独用来判断好坏）")

    print(f"\n===== 固定样例：{DEMO} =====")
    raw = generate(tok, policy, DEMO)
    print(f"  生成   : {raw!r}")
    print(f"  RM 打分: {reward_of(tok, reward, DEMO, raw):+.2f}")

    if failures:
        print(f"\n===== 未完全匹配的样例（共 {len(failures)} 条，展示前 8 条）=====")
        for user, tgt, raw, pred in failures[:8]:
            print(f"\nInput      : {user}")
            print(f"Target     : {json.dumps(tgt, ensure_ascii=False)}")
            print(f"Prediction : {json.dumps(pred, ensure_ascii=False) if pred else raw!r}")


if __name__ == "__main__":
    main()
