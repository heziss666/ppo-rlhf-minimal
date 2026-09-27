"""最小 PPO-RLHF：SFT 策略 + Reward Model + Value Model + TRL PPOTrainer。

SFT Policy -> 生成回答 -> RM 打分 -> Critic 估值 -> GAE -> PPO -> 策略更偏向高 reward 回答

四个模型的角色（TRL 接口里的名字）：
  model         = Policy（Actor），Base+SFT adapter 已 merge，PPO 阶段只训练新的 LoRA
  ref_model     = Reference（KL 约束）。传 None，TRL 用 disable_adapter() 现算，即「关掉 PPO LoRA」的 SFT 模型
  reward_model  = Reward Model，冻结，只当裁判
  value_model   = Critic，用同一个 RM 初始化，只训练 score head

运行：python train.py
小规模自检：python train.py --prompts 16 --epochs 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")  # trl.experimental 的稳定性警告

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.ppo import PPOConfig, PPOTrainer

HERE = Path(__file__).parent
BASE = "Qwen/Qwen3-0.6B"
SFT_ADAPTER = HERE.parent / "sft_tool_call" / "output"
RM_ADAPTER = HERE.parent / "reward-model-minimal" / "output"
DATA = HERE / "prompts.jsonl"
OUTPUT = HERE / "output"
SYS = "你是一个工具调用助手。根据用户需求选择工具并提取参数，只输出合法JSON，不要解释。"
CHAT_KWARGS = {"enable_thinking": False}


def seq_cls_from_rm(dtype):
    """Base seq-classification + RM adapter，merge 后 score head 就是训练好的 reward head。"""
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, dtype=dtype)
    return PeftModel.from_pretrained(base, str(RM_ADAPTER)).merge_and_unload()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=0, help="只用前 N 条 prompt（0=全部），小规模自检用")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--vf-coef", type=float, default=0.1, help="critic loss 权重；0.1 是 PPO 常见默认值")
    a = ap.parse_args()

    # ---- 数据：PPO 只需要 prompt，没有 completion ----
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    ds = load_dataset("json", data_files=str(DATA), split="train")
    if a.prompts:
        ds = ds.select(range(min(a.prompts, len(ds))))

    def to_ids(row):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": row["prompt"]}]
        return {"input_ids": tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, **CHAT_KWARGS)["input_ids"]}

    ds = ds.map(to_ids, remove_columns=ds.column_names)
    print(f"prompts: {len(ds)} 条 | 第一条 token 数 = {len(ds[0]['input_ids'])}")

    # ---- Policy / Actor：Base + SFT adapter 合并成完整 SFT 模型 ----
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16), str(SFT_ADAPTER)
    ).merge_and_unload()

    # ---- Reward Model：冻结，只当裁判 ----
    reward = seq_cls_from_rm(torch.bfloat16)
    reward.config.pad_token_id = tok.pad_token_id
    reward.requires_grad_(False)

    # ---- Value Model / Critic：同一个 RM 初始化，只训练 score head ----
    value = seq_cls_from_rm(torch.bfloat16)
    value.config.pad_token_id = tok.pad_token_id
    for name, p in value.named_parameters():
        p.requires_grad = "score" in name

    print("\n[四个模型]")
    print(f"  Policy (Actor)     : Base + SFT adapter(merged) + 新的 PPO LoRA   <- 训练")
    print(f"  Reference          : ref_model=None，TRL 用 disable_adapter() 现算 SFT 模型  <- 冻结")
    print(f"  Reward Model       : Base seq-cls + RM adapter                    <- 冻结，只当裁判")
    print(f"  Value (Critic)     : 同一个 RM 初始化，只训练 score head           <- 训练")
    critic_train = sum(p.numel() for p in value.parameters() if p.requires_grad)
    print(f"  Critic 可训练参数   : {critic_train:,}（只含 score head）")

    # ---- PPO 配置 ----
    args = PPOConfig(
        output_dir=str(OUTPUT),
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=a.epochs,
        num_ppo_epochs=4,
        num_mini_batches=1,
        response_length=64,
        temperature=0.7,
        stop_token="eos",
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=a.vf_coef,
        kl_coef=0.05,
        whiten_rewards=False,
        save_strategy="no",
        report_to="none",
        num_sample_generations=0,
    )

    # ---- 组装：model / ref_model / reward_model / value_model ----
    trainer = PPOTrainer(
        args=args,
        processing_class=tok,
        model=policy,
        ref_model=None,
        reward_model=reward,
        value_model=value,
        train_dataset=ds,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ),
    )

    trainer.train()
    trainer.save_model(str(OUTPUT))
    print(f"\nPPO adapter 已保存到 {OUTPUT}")


if __name__ == "__main__":
    main()
