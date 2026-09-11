# Distributed Training 学习计划（基于 torchtitan，以 DeepSeek V4 为例）

> 目标框架：[`torchtitan`](https://github.com/pytorch/torchtitan) —— PyTorch 官方出品的"原生"大模型分布式训练平台。
> 论文：[arXiv:2410.06511](https://arxiv.org/abs/2410.06511)（已被 ICLR 2025 接收）
> 示例模型：**DeepSeek V4**（`torchtitan/models/deepseek_v4/`）—— 一个原生 MoE + 稀疏 Attention 模型，天然覆盖 DP/TP/PP/CP/EP 全部并行维度，非常适合作为学习并行的统一示例。

## 0. 这个框架支持哪些并行方式？（先建立全局认知）

torchtitan 的核心卖点是 **多维可组合并行（multi-dimensional composable parallelism）**：各种并行方式都通过一个统一的 `DeviceMesh` 组织，可以任意组合叠加。对应源码入口是
[`torchtitan/distributed/parallel_dims.py`](../torchtitan/distributed/parallel_dims.py) 里的 `MeshAxisName`，它定义了六个网格轴：

| 轴名 | 全称 | 作用 | 对应模块 |
|---|---|---|---|
| `dp_replicate` | Data Parallel（复制） | 传统 DDP，梯度 all-reduce | `torch.distributed` DDP |
| `dp_shard` | FSDP2 分片 | 参数/梯度/优化器状态按 per-parameter 切片 | `torchtitan/distributed/fsdp.py` |
| `tp` | Tensor Parallel | 单层内部矩阵乘法切分（含 Async TP） | `torchtitan/distributed/tensor_parallel.py` |
| `pp` | Pipeline Parallel | 按层切 stage，micro-batch 流水线 | `torchtitan/distributed/pipeline_parallel.py` |
| `cp` | Context Parallel | 长序列按 sequence 维度切分 | `torchtitan/distributed/context_parallel/` |
| `ep` / `efsdp` | Expert Parallel（MoE） | 专家分布到不同 rank + 专家参数 FSDP | `torchtitan/distributed/deepep/`, `minimal_async_ep/` |

此外还有 `dp_replicate + dp_shard` 组合出的 **HSDP**（Hybrid Sharded Data Parallel），以及围绕这些并行的一整套配套能力：Float8/MXFP8 低精度训练、`torch.compile`、Activation Checkpointing、Distributed Checkpointing（DCP，可异步）、TorchFT 容错训练、`flex_shard`（优化器状态动态 reshard，含 Muon 优化器分布式实现）。

**一句话总结**：torchtitan 支持 **DP（DDP / FSDP2 / HSDP）+ TP（含 Async TP）+ PP + CP + EP** 五种并行原语的任意 N-D 组合。**DeepSeek V4** 作为一个稀疏 Attention + MoE 模型，是仓库里把这五种并行原语用得最全的模型之一，所以本计划全程用它做示例。

**DeepSeek V4 模型速览**（[`torchtitan/models/deepseek_v4/README.md`](../torchtitan/models/deepseek_v4/README.md)）：
- `model.py`：decoder 模型与 transformer block 定义
- `attention.py`：稀疏 Attention 变体（滑窗、压缩注意力、压缩稀疏注意力）
- `compressor.py`：KV 压缩与稀疏 index 选择
- `moe.py`：MoE 路由与专家实现
- `sharding.py`：**声明式**分片配置（TP/SP/EP/FSDP/DTensor/`spmd_types` 都在这一个文件里定义，对应其它模型里的 `parallelize.py`）
- `mtp.py`：Multi-Token Prediction 分支
- 已注册配置：`deepseek_v4_debugmodel` / `deepseek_v4_mtp_debugmodel` / `deepseek_v4_flash` / `deepseek_v4_pro`

同时仓库提供了一个现成的单卡冒烟测试脚本，全程会用到它：
[`scripts/run_deepseek_debug.sh`](../scripts/run_deepseek_debug.sh)（内部固定 `MODULE=deepseek_v4 CONFIG=deepseek_v4_debugmodel`，再转发到 `run_train.sh`）。

---

## 学习路径总览

| 阶段 | 内容 | 预计用时 | 产出 |
|---|---|---|---|
| 第 0 周 | [分布式训练基础概念](./week0-distributed-fundamentals.md) | 2-3 天 | 能解释 collective ops、理解显存瓶颈来源 |
| 第 1 周 | [数据并行：DDP → FSDP2 → HSDP](./week1-data-parallel.md) | 3-4 天 | 单机多卡跑通 DeepSeek V4 + FSDP2，画出显存/通信 timeline |
| 第 2 周 | 张量并行 TP（含 Async TP） | 3-4 天 | 理解 DTensor，跑通 DeepSeek V4 的 TP+FSDP 2D 并行 |
| 第 3 周 | 流水线并行 PP | 4-5 天 | 理解 schedule（1F1B / zero-bubble），跑通 PP+DP |
| 第 4 周 | 上下文并行 CP（长序列） | 2-3 天 | 理解 Ring Attention / all-to-all，跑通长序列训练 |
| 第 5 周 | 专家并行 EP（MoE） | 3-4 天 | 理解 token routing + all-to-all，跑通 DeepSeek V4 的 MoE 专家并行 |
| 第 6 周 | 组合 4D/5D 并行 + 性能分析 | 4-5 天 | FSDP2+TP+EP(+PP/CP) 全组合，测 MFU |
| 第 7 周（进阶） | 精度/编译/容错/checkpoint | 3-5 天 | Float8/MXFP8、torch.compile、DCP、TorchFT |

---

## 第 0 周：分布式训练基础

> 详细展开版（三堵墙、collective 通信原语详解、显存估算实例、两个坑的原理剖析、自测清单）：[`week0-distributed-fundamentals.md`](./week0-distributed-fundamentals.md)

**要搞懂的概念：**
- 为什么需要分布式训练：显存墙（参数+梯度+优化器状态+激活值）、算力墙、通信墙。
- Collective 通信原语：`all-reduce`、`all-gather`、`reduce-scatter`、`all-to-all`、`broadcast`，以及它们的通信量/延迟特征。
- 并行方式的分类：
  - **模型无关的并行**（数据并行 DP）：复制模型，切分数据。
  - **模型内部的并行**（TP / PP / CP / EP）：切分模型本身的计算图。
- 显存占用估算（参数量 × 精度字节数 × (1 + 梯度 + Adam 状态×2) 的经验公式）。

**推荐阅读：**
- torchtitan 论文 [§2 Background](https://arxiv.org/abs/2410.06511)
- PyTorch 官方 [Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)

**动手：** 先用仓库自带的单卡冒烟测试脚本跑通 DeepSeek V4 debug 模型，熟悉仓库结构和 CLI：
```bash
./scripts/run_deepseek_debug.sh
```
读一遍脚本本身（只有 60 行），会顺带学到两个很实用的坑：
1. `deepseek_v4_debugmodel` 默认的 attention mask 是 O(n^2) 的稠密矩阵，序列/microbatch 太大会直接把单卡显存打爆，脚本里用 `SEQ_LEN` 收敛到 4096 来避免。
2. CUDA Graph 默认开启，但 `expert_parallel_degree=1`（即没开 EP）时标准 MoE dispatcher 里有一次非法的 CPU<->CUDA 拷贝，所以脚本默认 `CUDA_GRAPHS=0` 关掉 —— 这也提前预告了第 5 周 EP 部分要处理的问题。

---

## 第 1 周：数据并行 —— DDP → FSDP2 → HSDP

> 详细展开版（DDP/ZeRO/FSDP2/HSDP 原理剖析、`apply_fsdp_to_decoder` 代码精读、DeepSeek V4 的 `efsdp` 专家分片、单卡用 `fake_backend` 实测显存分片效果、自测清单）：[`week1-data-parallel.md`](./week1-data-parallel.md)

**要搞懂的概念：**
- DDP：每张卡持有完整模型副本，反向传播后对梯度做 all-reduce。
- ZeRO 思想（FSDP 的理论基础）：把参数/梯度/优化器状态分片到各 rank，通信换显存。
- **FSDP2**（torchtitan 默认）：per-parameter sharding，用 `DTensor` 表示分片参数；forward 前 all-gather 参数，backward 后 reduce-scatter 梯度。
- **HSDP**：`dp_replicate × dp_shard` 二维组合——组内 FSDP 分片、组间 DDP 复制，兼顾大规模训练的容错和通信效率（常用于多机场景，机内分片、机间复制）。

**精读代码：**
- [`docs/fsdp.md`](../docs/fsdp.md)
- [`torchtitan/distributed/fsdp.py`](../torchtitan/distributed/fsdp.py)
- [`torchtitan/models/deepseek_v4/sharding.py`](../torchtitan/models/deepseek_v4/sharding.py)：看 FSDP 部分是怎么应用到 DeepSeek V4 的 dense 层和 MoE 专家层上的（专家参数的 FSDP 分片和普通层不完全一样，留意 `efsdp` 相关逻辑）。

**动手：**
```bash
# 纯 FSDP（dp_shard=4）
NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.data_parallel_shard_degree 4

# HSDP：2 组复制 x 每组 2 卡分片
NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.data_parallel_replicate_degree 2 \
  --parallelism.data_parallel_shard_degree 2
```
用 [`docs/debugging.md`](../docs/debugging.md) 里的 profiling 工具抓一次 trace，观察 all-gather / reduce-scatter 在 timeline 上的位置。

---

## 第 2 周：张量并行 TP（Tensor Parallel）

**要搞懂的概念：**
- Megatron-LM 风格 TP：把 Attention 的 QKV/输出投影、MLP 的两个线性层按列/行切分，中间插入一次 all-reduce（或用 DTensor 自动处理为 `Partial → Replicate`）。
- `DTensor`（PyTorch 原生分布式 Tensor 抽象）：理解 `Shard(dim)` / `Replicate()` / `Partial()` placement。
- **Async TP**：把 TP 的通信与矩阵乘计算做 overlap（micro-pipeline all-gather/matmul），减少 TP 带来的气泡。
- TP 的通信量随 TP 并行度线性增长，一般只在**单机内**（NVLink 高带宽域）使用，很少跨机。
- DeepSeek V4 的 Attention 是稀疏/压缩注意力（见 `attention.py`、`compressor.py`），TP 切分方式和标准 Attention 不完全一样，`sharding.py` 里的 `set_deepseek_v4_attention_sharding` / `set_compressor_sharding` / `set_indexer_sharding` 就是专门为此写的分片规则，值得精读。

**精读代码：**
- [`torchtitan/distributed/tensor_parallel.py`](../torchtitan/distributed/tensor_parallel.py)
- [`torchtitan/models/deepseek_v4/sharding.py`](../torchtitan/models/deepseek_v4/sharding.py) 中 `set_deepseek_v4_attention_sharding`、`set_compressor_sharding`、`set_indexer_sharding`
- PyTorch [DTensor / TP 文档](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Async TP 讨论帖](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)

**动手：**
```bash
# 2D 并行：FSDP(dp_shard=2) + TP(tp=2)
NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 2

# 打开 async TP 对比吞吐/MFU 差异
```

---

## 第 3 周：流水线并行 PP（Pipeline Parallel）

**要搞懂的概念：**
- 按层把模型切成多个 stage，每个 stage 跑在不同 rank 上，用 micro-batch 排出流水线，减少"气泡"（bubble）。
- 经典调度：GPipe、1F1B（One-Forward-One-Backward）、Interleaved 1F1B、Zero-Bubble。
- torchtitan 如何让模型"pipeline-friendly"（见 [`docs/composability.md`](../docs/composability.md) 里 "Making the model pipeline friendly" 一节：ModuleDict 保留 FQN、seed checkpoint 初始化方案）。
- DeepSeek V4 还有一个 **MTP（Multi-Token Prediction）** 分支（`mtp.py`，对应 `deepseek_v4_mtp_debugmodel` 配置），切 PP stage 时要想清楚 MTP 头挂在哪个 stage 上，这是理解"非纯 for 循环 decoder"模型做 PP 切分的一个好例子。
- PP 与 DP/TP 的组合关系：PP 通常放在跨机维度（通信量小但延迟敏感）。

**精读代码：**
- [`torchtitan/distributed/pipeline_parallel.py`](../torchtitan/distributed/pipeline_parallel.py)
- [`torchtitan/models/deepseek_v4/mtp.py`](../torchtitan/models/deepseek_v4/mtp.py)
- [Zero-bubble PP 讨论帖](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
- [`docs/composability.md`](../docs/composability.md)（seed checkpoint 初始化、moduledict 技巧）

**动手：**
```bash
# 3D 并行：DP + TP + PP
NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 2

# 对比开启 MTP 分支后的 PP 切分
NGPU=4 MODULE=deepseek_v4 CONFIG=deepseek_v4_mtp_debugmodel ./run_train.sh \
  --parallelism.pipeline_parallel_degree 2 \
  --dump_folder ./outputs/deepseek_mtp_pp
```
对比不同 PP schedule（可在 config 里切换）对气泡率、显存的影响。

---

## 第 4 周：上下文并行 CP（Context Parallel）

**要搞懂的概念：**
- 长序列训练时，即使 TP/PP 都用上，单条序列的激活值仍可能爆显存 —— CP 把 **序列维度**切分到多个 rank。
- 核心机制：Ring Attention / all-to-all，每个 rank 只算自己那一段 Q，但需要看到全部 K/V（通过环形通信逐步交换）。
- CP 与因果 mask 的负载均衡问题（causal attention 前面 token 计算量小，需要 load-balance 切分策略）。
- **重点结合第 0 周踩过的坑**：`deepseek_v4/attention.py::_build_block_mask` 会构造一个 `[1, stream_len, stream_len]` 的稠密 mask，序列越长显存占用是 O(n^2) 增长的。这正是 CP 要解决的问题之一 —— 开 CP 之后，每个 rank 只需要处理切分后的一段序列，直观感受 CP 如何压低这个 O(n^2) 的峰值显存。

**精读代码：**
- [`torchtitan/distributed/context_parallel/api.py`](../torchtitan/distributed/context_parallel/api.py)
- [`torchtitan/models/deepseek_v4/attention.py`](../torchtitan/models/deepseek_v4/attention.py)（`_build_block_mask` 及稀疏 Attention 变体）
- [CP 长序列讨论帖（1M context）](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082)

**动手：**
```bash
# 不开 CP，逐步加大 SEQ_LEN，找到单卡 OOM 边界
SEQ_LEN=8192 NGPU=1 ./scripts/run_deepseek_debug.sh

# 开 CP=4 后再跑相同/更大的 SEQ_LEN，对比显存曲线
SEQ_LEN=16384 NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.context_parallel_degree 4
```

---

## 第 5 周：专家并行 EP（Expert Parallel，针对 MoE 模型）

**要搞懂的概念：**
- DeepSeek V4 是原生 MoE 模型（`moe.py`），不同 token 被路由到不同"专家"（expert）子网络。
- EP 把不同专家分布到不同 rank，token 通过 **all-to-all** 被 dispatch 到对应专家所在的 rank，计算完再 all-to-all 收回。
- `efsdp`：专家参数本身还可以再做 FSDP 分片（专家数量大时单卡放不下）。
- torchtitan 里两套 EP 通信实现：
  - [`deepep/`](../torchtitan/distributed/deepep/)：对接 DeepEP（含 `hybridep.py`，即 HybridEP）。
  - [`minimal_async_ep/`](../torchtitan/distributed/minimal_async_ep/)：一个更"minimal"的异步 EP 参考实现，适合学习原理。
- 回顾第 0 周踩的坑：`expert_parallel_degree=1`（即不开 EP）时，标准 MoE token dispatcher 在 CUDA Graph capture 期间会做一次非法的 CPU<->CUDA 拷贝，所以 `run_deepseek_debug.sh` 默认关闭了 CUDA Graph。这周开 EP 之后，可以反过来验证：EP≥2 时使用 graph-safe 的 dispatcher，是否能重新打开 `CUDA_GRAPHS=1`。

**精读代码：**
建议**先看 `minimal_async_ep`**（代码更小、更适合学习原理），再看 `deepep`（生产级、性能更优）。
- [`torchtitan/models/deepseek_v4/moe.py`](../torchtitan/models/deepseek_v4/moe.py)：MoE 路由与专家实现
- [`torchtitan/models/deepseek_v4/sharding.py`](../torchtitan/models/deepseek_v4/sharding.py)：EP 相关的分片规则

**动手（复刻并扩展仓库 README 里的官方 smoke test）：**
```bash
# 官方 smoke test：FSDP2(2) + TP(2) + EP(2)
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 MODULE=deepseek_v4 CONFIG=deepseek_v4_debugmodel ./run_train.sh \
  --training.steps 1 \
  --metrics.log_freq 1 \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.expert_parallel_degree 2

# 用 debug 脚本跑更多步，并尝试重新打开 CUDA Graph
CUDA_GRAPHS=1 NGPU=4 STEPS=20 ./scripts/run_deepseek_debug.sh \
  --parallelism.expert_parallel_degree 2
```
可参考 [`torchtitan/models/deepseek_v4/README.md`](../torchtitan/models/deepseek_v4/README.md) 的 "Smoke Test" / "Status" 章节，以及 [`docs/release.md`](../docs/release.md) 里 "MoE FSDP+TP+EP+CP" 组合测试、DeepSeek V3 用 HybridEP 的验证案例。

---

## 第 6 周：组合 4D/5D 并行 + 性能分析

**目标**：能自己根据"多少张卡、什么模型、什么带宽拓扑"设计一套并行方案，并用指标验证。

**要搞懂的概念：**
- 并行度分配的经验法则：TP ≤ 单机 GPU 数（走 NVLink）；PP 放跨机维度（通信量小）；DP(FSDP) 尽量吃满剩余卡数；CP 按序列长度需求开；EP 按专家数/机内外拓扑开。
- 关键指标：**MFU**（Model FLOPs Utilization）、tokens/sec、显存占用，torchtitan 训练时会直接打印/记录到 Tensorboard/W&B（见 [`docs/metrics.md`](../docs/metrics.md)）。
- 用 [`torchtitan/distributed/parallel_dims.py`](../torchtitan/distributed/parallel_dims.py) 理解 `DeviceMesh` 是怎么把这些轴组织成一个多维网格的，并结合 `sharding.py` 的 `set_deepseek_v4_sharding_config` 看这些轴具体是怎么"接"到 DeepSeek V4 各层上的。
- 试试 `--parallelism.spmd_backend spmd_types`（DeepSeek V4 README 里提到的可选后端），对比默认 `partial_dtensor` 后端的行为/性能差异。

**动手：**
```bash
# 在 4 卡范围内组合 FSDP + TP + EP（若卡数够，可再加 PP/CP）
NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.expert_parallel_degree 2 \
  --parallelism.spmd_backend spmd_types
```
1. 固定总卡数，尝试 2-3 组不同的并行度切分（如 FSDP4 vs FSDP2+TP2 vs TP2+EP2），对比 MFU / 吞吐差异，写一份小结。
2. 用 [`docs/debugging.md`](../docs/debugging.md) 里的 Flight Recorder / profiler 工具定位一次通信瓶颈。
3. 如果有更大规模的卡（>=8），尝试跑到官方 smoke test 之外的组合，比如再加一维 CP，验证 `deepseek_v4/attention.py` 的稀疏 mask 在 CP 切分下是否正确 load-balance。

---

## 第 7 周（进阶）：精度、编译、容错、Checkpoint

这些不是"并行方式"，但都是让上述并行方式在真实大规模训练里能跑起来、跑得快、跑得稳的关键配套能力：

- **`torch.compile`**：对并行后的模型做图编译加速。
- **Float8 / MXFP8**：降低通信和计算的字节数，[`torchtitan/components/quantization/float8.md`](../torchtitan/components/quantization/float8.md)、[`mxfp8/README.md`](../torchtitan/components/quantization/mxfp8/README.md)。
- **Activation Checkpointing**（per-op selective / full）：用重计算换显存 —— 对 DeepSeek V4 这种自带 O(n^2) mask、显存敏感的模型尤其值得练习。
- **Distributed Checkpointing (DCP)**：[`docs/checkpoint.md`](../docs/checkpoint.md)，支持异步保存；DeepSeek V4 的 checkpoint key 映射见 [`state_dict_adapter.py`](../torchtitan/models/deepseek_v4/state_dict_adapter.py)。
- **TorchFT**：容错训练（节点故障时不整体重启）。
- **`flex_shard`**（含 `dist_muon.py`）：优化器状态的动态 reshard，以及 Muon 优化器的分布式实现——如果你对新优化器感兴趣可以顺带学习。

**推荐阅读顺序**：Float8 → Activation Checkpointing → DCP → torch.compile → TorchFT → flex_shard（由易到难）。

---

## 学习方法建议

1. **每个并行方式都遵循"读文档 → 读源码 → 改配置跑一次 → 抓 profile/trace 看懂通信"这个闭环**，不要只停留在概念层面。
2. torchtitan 的一大优势是"模型代码几乎不改就能套上各种并行"——精读 `torchtitan/models/deepseek_v4/sharding.py` 这种文件时，重点看**并行是怎么从模型代码里"解耦"出去的**，这是工程上最值得学的部分。
3. 善用 [`docs/debugging.md`](../docs/debugging.md)：CPU/GPU profiling、内存 profiling、Flight Recorder，是理解"并行到底在干什么"最直接的手段。
4. 如果没有 8 卡环境，`deepseek_v4_debugmodel` 配置 + `scripts/run_deepseek_debug.sh` + 2-4 张卡也能把 DP/TP/PP/CP/EP 的组合逻辑跑通（只是规模小），先把逻辑吃透，再去追求大规模。
5. 每周结束写一份 3-5 句话的小结（哪怕就是给自己看），强制自己把"理解"转成"能说清楚"。

## 参考资料清单

- torchtitan 论文：<https://arxiv.org/abs/2410.06511>
- torchtitan README（本仓库）：[`README.md`](../README.md)
- DeepSeek V4 模型说明：[`torchtitan/models/deepseek_v4/README.md`](../torchtitan/models/deepseek_v4/README.md)
- DeepSeek V4 单卡冒烟测试脚本：[`scripts/run_deepseek_debug.sh`](../scripts/run_deepseek_debug.sh)
- torchtitan 扩展/贡献指南：[`docs/extension.md`](../docs/extension.md)
- PyTorch 官方分布式总览：<https://pytorch.org/tutorials/beginner/dist_overview.html>
- PyTorch DTensor / TP 文档：<https://pytorch.org/docs/stable/distributed.tensor.parallel.html>
- GPU MODE 讲座（torchtitan）：<https://www.youtube.com/watch?v=VYWRjcUqW6w>
