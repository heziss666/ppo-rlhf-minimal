# ppo-rlhf-minimal

最小可运行的 **PPO-RLHF** 实验，把前面两个项目接起来：

```
sft-tool-call/output           ->  Policy / Actor  +  Reference
reward-model-minimal/output    ->  Reward Model   +  Value Model (Critic)
```

完整链路：

```
SFT Policy -> 生成回答 -> RM 打分 -> Critic 估值 -> GAE -> PPO -> 策略更偏向高 reward 回答
```

## 四个模型

| 模型 | 来源 | 是否更新 | 作用 |
|---|---|---|---|
| Policy / Actor | Base + SFT adapter（merge）再加一层 PPO LoRA | 训练 | 生成回答 |
| Reference | `ref_model=None`，TRL 用 `disable_adapter()` 现算 | 冻结 | KL 约束 |
| Reward Model | Base seq-cls + RM adapter | 冻结 | 只当裁判 |
| Value / Critic | 同一个 RM 初始化，只训练 `score` head | 训练 | 预测令牌级 V(s) |

Reference 没有单独占一份显存：PPO LoRA 是加在 merge 后的 SFT 模型上的，把这个 adapter 关掉就是 π_ref。

## 安装与运行

TRL 已经**彻底移除**了 `PPOTrainer`（1.14 顶层和 `experimental` 都没有，上游 main 也没有）。最后一个带它的是 **0.28.0**，位于 `trl.experimental.ppo`。

所以本项目用一个隔离 venv，复用 DL 环境的 torch / transformers，只换 trl：

```powershell
# 在已有 torch + transformers 的环境基础上建 venv
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install "trl==0.28.0" --no-deps
```

版本（跑通时实测）：`trl 0.28.0` / `transformers 5.14.1` / `peft 0.19.1` / `torch 2.13.0.dev+cu132`，单卡 RTX 5070 Ti Laptop。

```powershell
.\.venv\Scripts\python.exe evaluate.py --base              # PPO 之前的 SFT 策略（基线）
.\.venv\Scripts\python.exe train.py                        # PPO 训练，约 2 分钟
.\.venv\Scripts\python.exe evaluate.py --adapter ./output  # PPO 之后
```

`evaluate.py` 只依赖 transformers + peft，用主环境的 python 也能跑。

先小规模自检整条链：

```powershell
.\.venv\Scripts\python.exe train.py --prompts 16 --epochs 1
```

## 文件

| 文件 | 说明 |
|---|---|
| `prompts.jsonl` | 80 条 prompt，**只有 prompt 没有答案**，PPO 让策略自己 rollout |
| `test.jsonl` | 20 条 prompt + 标准答案，用于任务指标 |
| `train.py` | 组装 4 个模型 + PPOTrainer |
| `evaluate.py` | 4 项任务指标 + Average RM Reward + 固定样例 |
| `output/` | PPO 阶段学出来的 LoRA adapter |

## 结果

| 指标 | SFT（PPO 前） | PPO `vf_coef=0.1` | PPO `vf_coef=1.0` |
|---|---|---|---|
| JSON Valid | 90.0% | 100.0% | 95.0% |
| Tool Accuracy | 70.0% | 50.0% | 55.0% |
| Arguments Accuracy | 45.0% | 40.0% | 40.0% |
| Exact Match | 45.0% | **25.0%** | **30.0%** |
| Average RM Reward | 5.69 | 4.65 | 5.41 |

**两次 PPO 都让策略变差了，而且 RM reward 也一起下降。**这不是 reward hacking（reward 没有升），是策略真的漂移了。两次的命令分别是：

```powershell
.\.venv\Scripts\python.exe train.py                 # vf_coef 默认 0.1
.\.venv\Scripts\python.exe train.py --vf-coef 1.0   # 提高 critic 权重 10 倍
```

## 为什么没work（诊断）

训练日志里的四个证据：

| 现象 | 数值 | 含义 |
|---|---|---|
| `loss/value_avg` | 全程 9~13，**从不下降** | Critic 完全没拟合上，V(s) 预测离 return 差约 4.5 |
| `policy/entropy_avg` | ≈ 0.1 | SFT 策略已经极度自信，rollout 几乎没有探索 |
| `policy/approxkl_avg` | ≈ 0.0005 | 每次更新的策略位移极小 |
| `objective/kl` | 0 → 0.66 一路上升 | 策略在持续漂移，但没有换来 reward |

因果链：

```
Critic 学不动 (loss 卡在 10)
      ↓
TRL 的 PPO 用 critic-based GAE 算 advantage，没有 GRPO 那种组内相对基准
      ↓
advantage ≈ critic 的误差，而不是 reward 的真实差异
      ↓
策略梯度变成噪声，把策略推离 SFT 最优点
      ↓
准确率掉、RM reward 也掉
```

直接证据是失败样例的变化。SFT 的错误是「参数名不对」（`location` 代替 `city`），PPO 之后开始**发明不存在的工具名**：

```
SFT : {"tool":"calculator","arguments":{"expression":"38*17"}}   参数格式问题
PPO : {"tool":"math","arguments":{"expression":"38*17"}}         工具名是编的
PPO : {"tool":"提醒","arguments":{"提醒内容":"..."}}              中文工具名 + 中文参数键
```

这是「没有可靠优势信号时策略随机漂移」的典型表现。

**根本原因**：SFT 策略已经足够好（Exact Match 45%），而且近乎确定性。TRL PPO 每个 prompt 只采样一次，advantage 全靠 critic；critic 又是只训练 `score` head（1024 参数）的小模型，学的又是逐 token 的 V(s)，在 160 个 optimizer step 内根本拟合不上。

可能的后续方向（第一版都没做）：

1. **给 Critic 加 LoRA**，而不只是训 head —— 这是最直接的修法，Critic 现在容量太小
2. `whiten_rewards=True`，把 advantage 归一化到零均值，去掉系统性偏差
3. 提高 rollout `temperature` 制造真正的探索，让 reward 差异进入 advantage
4. 换 **GRPO**：它不需要 critic，用同一 prompt 的组内相对分数做 advantage，正好绕开这个瓶颈

## 核心代码位置

| 步骤 | 位置 |
|---|---|
| prompt -> chat template -> `input_ids` | `train.py:63` |
| Policy = Base + SFT adapter（merge） | `train.py:72` |
| Reward Model（冻结） | `train.py:75` |
| Value Model / Critic（只训 score head） | `train.py:80` |
| PPOConfig | `train.py:94` |
| PPOTrainer：`model` / `ref_model` / `reward_model` / `value_model` | `train.py:118` |
| rollout + GAE + PPO 更新 | `train.py:132` |
| 任务指标 + RM reward | `evaluate.py:139` |

GAE / PPO clipped objective 本身没有自己实现，在 TRL 内部：

```python
# trl/experimental/ppo/ppo_trainer.py
output, vpred_temp = forward(model, mb_query_responses, ...)      # 一次 forward 出 policy + value
ratio = torch.exp(new_logprobs - mb_logprobs)                     # PPO ratio
pg_losses = -mb_advantage * ratio
pg_losses2 = -mb_advantage * torch.clamp(ratio, 1-clip, 1+clip)
loss = masked_mean(torch.max(pg_losses, pg_losses2), ...) + vf_coef * vf_loss
```
