# llm2jev

[English](README.md)

把**任意对话模型**变成兼容 [Jev](https://docs.typesafe.ai/api) 的概率决策服务，可运行在
**SGLang、vLLM、原生 transformers 或 MLX（Apple 芯片）** 上，无需改动模型或推理引擎。

```
状态 + 问题 + 全部选项  ──一次 prefill──▶  同一位置上各选项标签的 logprob  ──▶  概率
```

- **每个问题只做一次前向**，无论有多少个选项（最多 255 个）。不生成、不采样。
- **所有选项放在同一个 prompt 里**，选项之间直接竞争（"以上都不是"可以依赖其他选项）。
- **标准 chat template，关闭 thinking。** 指令模型可零样本直接使用；微调模型使用同样的 prompt。
- **共享前缀只渲染一次。** 同一请求中的所有问题都以字节完全相同的文本开头，前缀先预热一次，其余部分由引擎的前缀缓存
  （SGLang radix / vLLM APC）命中。
- **状态中可包含图片、视频和音频**（SGLang、vLLM 和 transformers 后端），适用于多模态模型。

llm2jev 是对公开文档中 System One 通信格式的独立实现，与 TypeSafe 没有关联。

它在 0.5.0 之前叫 AnyJev。

## 快速开始

```bash
pip install llm2jev              # 客户端 + 服务端；推理引擎单独运行
pip install "llm2jev[hf,vision]" # + 进程内 transformers 后端（含视频和音频解码器）
pip install "llm2jev[mlx]"       # + 进程内 MLX 后端（Apple 芯片）

# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3.5-2B --port 30000
llm2jev --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000

# vLLM（--enable-scale-out 开启 tokens-in 接口，图片、视频和音频要用到）
vllm serve Qwen/Qwen3.5-2B --max-logprobs 256 --return-tokens-as-token-ids --enable-scale-out --port 8000
llm2jev --model Qwen/Qwen3.5-2B --backend vllm --url http://127.0.0.1:8000

# transformers，进程内参考实现（慢，无前缀缓存）
llm2jev --model Qwen/Qwen3-0.6B --backend hf

# Apple 芯片上的 MLX，进程内（仅文本；支持 HF 模型 id 或 mlx-community 量化模型）
llm2jev --model mlx-community/Qwen3-0.6B-4bit --backend mlx
```

在 Python 中直接调用，不启动 HTTP 服务：

```python
from transformers import AutoProcessor
from llm2jev import LLM2Jev
from llm2jev.backends import SGLang

jev = LLM2Jev(AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B"), SGLang("http://127.0.0.1:30000"))
jev(state, questions)  # {"refund": {"type": "noul", "noul": 0.97}, ...}
```

```bash
curl localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": [{"role": "system", "content": "You are a support assistant."},
            {"role": "user", "content": "I was charged twice. Please refund the duplicate."}],
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency":    {"type": "score",  "instructions": "How urgent is the request?",
                   "criteria": ["Routine", "Urgent", "Emergency"]}}}'
```

多媒体：在状态消息中放入 `{"type": "image", "image": "<路径 | https URL | data: URI>"}`，视频和音频同理，
分别用 `{"type": "video", "video": ...}` 和 `{"type": "audio", "audio": ...}`（也可用 OpenAI 风格的 `image_url` /
`video_url` / `audio_url`）。transformers 后端只能从路径或 URL 读取视频（`llm2jev[hf]` 自带视频和音频解码器）。
vLLM 上的音频需要 `vllm[audio]`。

## 分数如何计算

请求用模型自带的 chat template 渲染一次（`enable_thinking=False`）。问题放在最后一个 user 轮次中，assistant 轮次
预填为 `Answer:`：

```
<|im_start|>user
Evaluate the conversation or state above using the question below. ... reply with its label only.

Question: How urgent is the request?
Options:
A. 0: Routine
B. 1: Urgent
C. 2: Emergency<|im_end|>
<|im_start|>assistant
<think>

</think>

Answer:▸ 读取 ' A' ' B' ' C'
```

引擎返回该位置上每个标签 token 在全词表上的 logprob ℓᵢ（`max_new_tokens=1`，指定 token 的 logprob）。
然后在该问题的标签上计算 p = softmax(ℓ / T)：

| 类型   | 答案 |
|--------|------|
| noul   | `P(Yes)`。选项为 `A. Yes`、`B. No`。 |
| choice | argmax、完整分布，以及 `confidence = 1 − H(p)/log K` |
| score  | 期望等级 `Σ i·pᵢ`（等级从 0 开始）、分布和置信度 |

标签依次为 `A`–`Z`，之后是双字母标签（`AA`、`AB`……）。启动时会**对照真实的 prompt 结尾检查**每个标签，
确保它在该位置恰好是一个新 token。因此模型在 `Answer:` 之后写 `'A'` 还是 `' A'`，由分词器决定，而不是靠猜。

`--temperature` 只缩放分布，不会改变 argmax。如需校准后的概率，可在留出数据上全局拟合一次。

## 状态

一致性检查 = `scripts/parity.py`：引擎与 transformers 参考实现对同样渲染好的 prompt 打分。每个问题的 argmax 必须一致，
概率差不超过 0.03。

| 后端 | 文本 | 图片 | 视频 | 音频 |
|---|---|---|---|---|
| transformers | 参考（Qwen3-0.6B、Qwen3.5-2B、Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B、Qwen2-Audio-7B） | 参考（Qwen3.5-2B、Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B） | 参考（Qwen3.5-2B、Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B）⁶ | 参考（Qwen2-Audio-7B） |
| SGLang 0.5.18 | 一致 ✓ Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B ⁴ | 一致 ✓ Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B | 一致 ✓ Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B | – |
| SGLang 0.5.9 | 一致 ✓ Qwen3-0.6B、Qwen3.5-2B、Qwen2-Audio-7B | 一致 ✓ Qwen3.5-2B | 一致 ✓ Qwen3.5-2B ¹ | 仅 30 秒片段 ² |
| vLLM 0.30.0 | 一致 ✓ Qwen3-0.6B、Qwen3.5-2B、Qwen2-Audio-7B | 一致 ✓ Qwen3.5-2B | 一致 ✓ Qwen3.5-2B | 一致 ✓ Qwen2-Audio-7B |
| vLLM 0.26.0 | 一致 ✓ Qwen3.6-27B、Qwen3.6-35B-A3B ⁴ | ✗ ⁵ | ✗ ⁵ | – |
| MLX (mlx-lm) | 与 transformers 一致 ✓ Qwen3-0.6B ³ | – | – | – |

¹ 在容差之内，但并非逐字节一致。对 Qwen3.5 视频，SGLang 去掉了模板中每帧块外层的
`<|vision_start|>…<|vision_end|>`（2 个 token），而 transformers 和 vLLM 保留了它们。SGLang 还会放大小尺寸帧
（128 px 变为约 320 px），所以一致性测试用的视频是 320 px。
² SGLang 0.5.9 运行 Qwen2-Audio 编码器时没有使用特征 attention mask，因此短于 Whisper 30 秒窗口的片段会关注到
零填充部分。2 秒片段的概率偏差为 0.20，30 秒片段则一致。音频请使用 vLLM 或 transformers。
³ `test_mlx_matches_hf`，在 mlx CPU 版本上运行。无前缀缓存：每个问题都是一次完整 prefill。
⁴ Qwen3.6（27B dense、35B-A3B MoE），TP 2 和 TP 4。SGLang 0.5.9 能加载 Qwen3.6，但打分是错的（文本概率偏差最高
0.8），请使用 0.5.18。35B-A3B 上所有 argmax 一致，但有一道接近平局的题（0.77 对 0.22）在 SGLang、vLLM、transformers
任意两者之间相差 0.05，这是 bf16 MoE 路由噪声，MoE 模型的一致性检查请用 `--tol 0.06`。Qwen3.8-27B 需加 `--enable-fp32-lm-head` 才能通过（bf16 head 下有一题为 0.031）。
⁵ vLLM 0.26 的 `/inference/v1/generate` 接受 `content_parts`，但会忽略其中的多媒体。多媒体请使用 vLLM 0.30+。
⁶ transformers 5 用 torchcodec 解码视频，失败时回退到 `torchvision.io.read_video`，而 torchvision 0.26 已删除该函数。
请安装与 torch 匹配的 torchcodec（torch 2.11 对应 0.11）。

0.01–0.03 的概率差异来自 bf16 kernel 噪声；分词完全一致。对 SGLang，`--enable-fp32-lm-head` 大约能把差距减半。
vLLM 把纯文本 prompt 发送到 `/v1/completions`，带多媒体的 prompt 则以精确的 token id 加上多媒体数据发送到
`/inference/v1/generate`，由引擎自己展开占位符。无法关闭 thinking 的模型需要一个会闭合 think 块的模板。

```bash
python scripts/parity.py --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000
```

### Prompt 风格

`--prompt chat`（默认）使用模型的 chat template，关闭 thinking。适用于指令模型，新模型的训练也建议用它。

`--prompt jevlm` 用于在不带 chat template 的原始补全 prompt 上微调的模型：
`State: … Question: … Options: A. … Answer with the letter of the best option.\nAnswer:`。仅支持文本。

## 用于强化学习

对每个问题，读出的是选项上的归一化分布 π(option | state)。它来自一次 prefill，运行在 RL 框架做 rollout 时已在使用的
同一批引擎上，因此可以直接作为决策策略。训练端请使用相同的 `llm2jev.prompt.render` 和标签 id，
让 rollout 和 learner 对同一个 logit 打分。

## 测试

```bash
pytest tests                    # 快速：假后端 + 真实分词器
LLM2JEV_SLOW=1 pytest tests      # + 通过 transformers 后端跑真实模型
```
