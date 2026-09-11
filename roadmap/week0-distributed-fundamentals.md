# 第 0 周：分布式训练基础概念详解

> 本文是 [`distributed-training-roadmap.md`](./distributed-training-roadmap.md) 第 0 周内容的展开讲解。
> 目标：搞懂"为什么需要分布式训练"、"通信原语在做什么"、"显存都花在哪"，为后面每一周具体的并行方式（DP/TP/PP/CP/EP）打底。
> 示例模型统一用仓库自带的 `deepseek_v4_debugmodel`（`torchtitan/models/deepseek_v4/`），配合单卡脚本 [`scripts/run_deepseek_debug.sh`](../scripts/run_deepseek_debug.sh)。

---

## 1. 为什么需要分布式训练：三堵墙

单卡训练大模型会先后撞上三堵墙，理解这三堵墙的成因，才能理解每种并行方式究竟"用什么换什么"。

### 1.1 显存墙（Memory Wall）

一次训练迭代，GPU 显存主要被四类东西占用：

1. **模型参数**（Parameters）
2. **梯度**（Gradients）—— 和参数同形状
3. **优化器状态**（Optimizer States）—— AdamW 需要一阶矩 `m` 和二阶矩 `v`，各和参数同形状
4. **激活值**（Activations）—— 前向过程中为了反向传播而必须保留的中间张量，随 `batch_size × seq_len × hidden_dim × n_layers` 增长

**经验公式（前三项，静态显存）：**

```
静态显存 ≈ 参数量 × 精度字节数 × (1 + 梯度 + Adam状态×2)
```

- 纯 fp32 训练：`1 (参数) + 1 (梯度) + 2 (Adam m,v) = 4`，每参数 `4 bytes × 4 = 16 bytes`
- 若用 Adam 的方差版本（如混合精度下 optimizer states 用 fp32、参数用 bf16），公式里每一项的字节数可以不同，不能直接套统一倍数，要分项算

**这次对话里已经拿 `deepseek_v4_debugmodel` 实测验证过这套公式**，直接搬过来作为具体案例：

模型 `Model deepseek_v4 debugmodel size: 10,601,565 total parameters`，torchtitan 默认 `training.dtype = "float32"`（参数/梯度/优化器状态都是 fp32，4 bytes；`mixed_precision_param="bfloat16"` 只是 FSDP 计算时的临时 cast，不改变主存副本）：

```
10,601,565 × 4 bytes × (1 + 1 + 2) = 169,625,040 bytes ≈ 161.8 MiB
```

即参数+梯度+优化器状态总共约 **170 MiB**——但这只是"静态"部分，公式里没算激活值。

**激活值这部分，公式里的经验系数很难套一个通用倍数**，因为它取决于是否开激活检查点（Activation Checkpointing，第 7 周内容）、序列长度、有没有 attention 里的额外缓冲区。`deepseek_v4_debugmodel` 就是一个很典型的反例：`attention.py::_build_block_mask` 会为每层构造一个 `[1, seqlen, kv_len]` 的**稠密** int32 缓冲区（`selected_count = torch.zeros(bsz, seqlen, kv_len, dtype=torch.int32, ...)`），这是 O(n²) 增长的：

| `seqlen`（= `num_tokens_per_microbatch_per_dp_rank`） | 单层缓冲区大小 |
|---|---|
| 16384（debugmodel 默认，8 折叠成 131072 流）| `131072² × 4 bytes ≈ 64 GiB` |
| 4096（`run_deepseek_debug.sh` 里 `SEQ_LEN` 覆盖后）| `4096² × 4 bytes = 64 MiB` |

这解释了脚本里为什么要用 `SEQ_LEN=4096` 而不是让 debugmodel 用默认序列长度跑——序列长度对显存的影响不是线性的，是平方的，这是**理解显存瓶颈来源**里最容易被低估的一点。

**实测验证（`./scripts/run_deepseek_debug.sh`，单卡 RTX 5060 Ti）**：`nvidia-smi` 观察到峰值 `2360 MiB`，这跟下面的分项估算基本吻合：

| 项 | 估算 | 依据 |
|---|---|---|
| 参数+梯度+优化器状态（fp32） | ~170 MB | 上面的公式，精确算 |
| Attention mask/索引缓冲（4 层，SEQ_LEN=4096） | ~64-256 MB（瞬时） | `_build_block_mask` 精确算 |
| 前向激活（无 AC，4 层保留） | ~300-400 MB | 按张量形状粗估 |
| CUDA context + cuDNN/NCCL 初始化 + allocator 固定开销 | ~0.5-1.5 GB | 经验值，跟模型大小基本无关 |
| **合计** | **~1-2.5 GB** | 与实测 2360 MiB 吻合 |

**关键结论**：对这种量级的 debug 模型，**固定开销（CUDA/cuDNN/NCCL context）远大于模型本身**——这也是为什么"模型小、显存却没降到几十 MB"的原因。等模型规模真正变大（GB 级参数），固定开销占比会迅速被参数/梯度/优化器状态/激活值反超，公式才真正开始体现价值。

**显存墙的本质**：单卡显存有限（消费卡 16-24GB，数据中心卡 80-192GB），而参数量 × 16 bytes（fp32 全量）在几十亿参数规模就轻松突破单卡容量，更别提激活值。**所有并行方式，本质上都是在参数、梯度、优化器状态、激活值这四类东西里选一类或几类，切分到多张卡上**——这是理解后面每一周内容的统一视角：

| 并行方式 | 切分对象 |
|---|---|
| DP / FSDP | 数据切分（DDP）或参数+梯度+优化器状态切分（FSDP） |
| TP | 单层内部的权重矩阵切分 |
| PP | 按层切分模型（不同层放不同卡） |
| CP | 激活值按序列维度切分 |
| EP | MoE 专家参数切分到不同卡 |

### 1.2 算力墙（Compute Wall）

训练一个模型大致需要 `6 × 参数量 × token 数`（记作 `6ND`）FLOPs。单卡算力（peak FLOPs/s）是有限的，即使显存足够，**单卡把这么多 FLOPs 算完也需要一个下限时间**。多卡并行把总 FLOPs 分摊到多个计算单元上同时做，才能把 wall-clock 训练时间压下来——这是"为什么不仅要解决显存，还要解决速度"的原因，也是后面 **MFU（Model FLOPs Utilization）** 指标的意义：MFU 越高，说明你花的钱（GPU-小时）换来的有效算力越多。

#### `6ND` 是怎么来的

这个公式来自 OpenAI 的 Scaling Laws 论文（Kaplan et al. 2020）和 Chinchilla 论文（Hoffmann et al. 2022），`6` 这个系数不是拍脑袋定的，是从矩阵乘法的 FLOPs 计数一步步推出来的。

**第一步：矩阵乘法的 FLOPs 计数是基础。** 一个 `[m, n] × [n, k]` 的矩阵乘法需要 `m×n×k` 次乘法 + `m×n×k` 次加法，共 `2×m×n×k` FLOPs。Transformer 里绝大部分计算量都来自这种矩阵乘法（Attention 的 QKV/输出投影、MLP 的两个线性层），而一个线性层 `Linear(in_features=n, out_features=k)` 的权重矩阵形状是 `[n, k]`，也就是说这个层的**参数量**正好是 `n×k`（忽略 bias）。

**第二步：前向 ≈ `2 × 参数量` （每 token）。** 对单个 token 过一个线性层：输入形状 `[1, n]`，权重 `[n, k]`，矩阵乘法 FLOPs = `2×1×n×k = 2nk`，正好是这个层参数量 `nk` 的 2 倍。把模型里所有线性层的参数量加起来就是总参数量 `N`，所以一个 token 完整过一遍网络（前向）的 FLOPs ≈ `2N`。处理 `D` 个 token 就是 `2ND`。

**第三步：反向 ≈ 前向的 2 倍。** 反向传播要对每个矩阵乘法算两个梯度：对权重的梯度 `dL/dW = x^T @ dL/dy`（跟前向同规模的矩阵乘法，FLOPs 也是 `2nk`），以及对输入的梯度 `dL/dx = dL/dy @ W^T`（同样是同规模的矩阵乘法，FLOPs 也是 `2nk`）。所以反向要多算两遍同规模的乘法，反向 FLOPs ≈ `2 × 前向 = 4N`（每 token）。

**合计：** 前向 `2N` + 反向 `4N` = `6N`（每 token），处理 `D` 个 token 总计 `6ND`。

**这个公式的近似前提（局限性）：**
- 只算了矩阵乘法主导的部分，Attention 的 softmax、RMSNorm、RoPE 这些逐元素/归一化操作的 FLOPs 相对矩阵乘法可以忽略。
- **忽略了 Attention 分数矩阵 `Q@K^T` 和 `softmax@V` 这两步跟序列长度平方相关的项**（`O(seq_len² × head_dim)`）。当 `hidden_dim >> seq_len` 时这一项相对 `2N` 可以忽略，但**序列长度很长时（比如第 4 周 CP 要处理的长上下文场景）这一项会显著增大，`6ND` 会低估实际算力需求**——这也是长序列训练算力墙比公式估的更严重的原因。
- 这里的 `N` 通常指**非 embedding 参数量**（embedding 层是查表，不是矩阵乘法，FLOPs 占比很小，一般忽略不计）。

**联系 MFU：** 有了 `6ND`，就能算训练这个模型理论最少需要多久：

```
理论最短训练时间 = 6ND / (单卡峰值FLOPs × 卡数 × MFU)
```

`6ND` 是理论需要的算力，实际跑训练时 GPU 不可能 100% 时间都在做有效矩阵乘法（有通信等待、kernel launch overhead、显存搬运等），MFU 就是"实际有效算力 / 理论峰值算力"的比值，第 6 周性能分析部分会具体用到。

### 1.3 通信墙（Communication Wall）

一旦把计算/参数切分到多张卡，卡与卡之间就必须交换数据（同步梯度、交换激活值、交换专家输出……）。**通信不是免费的**：

- 通信有**带宽**上限（同机内 NVLink 几百 GB/s，跨机网络通常几十 GB/s 量级，差一个量级以上）
- 通信有**延迟**（尤其是小消息、跨机时延迟占主导）
- 如果通信不能和计算 **overlap**（重叠执行），就会变成纯粹的等待时间，白白拉长训练时间

这就是为什么：
- TP 的切分粒度要求高带宽（几乎只用在机内 NVLink 域）
- PP 只需要相邻 stage 之间传激活值，通信量小，适合跨机
- Async TP / FSDP 的 prefetch / EP 的 overlap 这些优化，本质都是在"想办法把通信藏到计算背后"

**一句话总结三堵墙的关系**：显存墙决定"能不能把模型放到这些卡上"，算力墙决定"理论最快多久能算完"，通信墙决定"实际能不能达到理论最快"。三者共同决定了你需要用什么并行方式、什么并行度组合。

---

## 2. Collective 通信原语详解

分布式训练里几乎所有跨卡数据交换都能归到几个标准的 **collective communication** 原语上（来自 MPI 传统，PyTorch `torch.distributed` 直接复用了这套术语）。理解这几个原语的**输入输出形状**和**通信量级**，比记住"FSDP 用 all-gather"这种结论更重要——因为看懂 profiler trace 里的通信 op 就是靠这个。

约定：`N` 个 rank，每个 rank 起始持有大小为 `data` 的本地数据。

### 2.1 Broadcast（广播）

- **语义**：1 个 rank（root）把自己的完整数据发给所有其他 rank。
- **形状**：root 输入 `data`，所有 rank 输出都是这份 `data`（全量复制）。
- **通信量**：树形算法约 `O(log N)` 次跳，总带宽开销约 `data × (N-1)/N`（用 scatter+all-gather 实现时）。
- **典型场景**：训练开始时把 rank 0 的初始参数广播给所有 rank（保证初始化一致），或广播随机种子。

```
root: [data]  --broadcast-->  rank0:[data] rank1:[data] rank2:[data] rank3:[data]
```

### 2.2 All-Reduce（全局归约）

- **语义**：所有 rank 各自的 `data` 做一次 reduce 运算（通常是求和/求平均），**每个 rank 都拿到完整的归约结果**。
- **形状**：输入每 rank 一份 `data`（同形状），输出每 rank 一份**同形状**的归约结果。
- **通信量**：Ring All-Reduce 算法下，每个 rank 发送+接收总量约 `2 × data × (N-1)/N`——这是带宽最优的实现方式，PyTorch NCCL 后端默认就是 ring 算法（大消息时）或 tree 算法（小消息时）。
- **典型场景**：**DDP** 的梯度同步——每张卡算出完整梯度后，all-reduce 得到平均梯度，所有卡拿到同一份梯度去更新参数。

```
rank0:[g0]  rank1:[g1]  rank2:[g2]  rank3:[g3]
        \-------- all-reduce (sum) --------/
rank0:[g0+g1+g2+g3]  rank1:[同上]  rank2:[同上]  rank3:[同上]
```

### 2.3 Reduce-Scatter（归约后分片）

- **语义**：所有 rank 的 `data` 做归约，但**结果按 rank 切成 N 份，每个 rank 只拿到属于自己的那一份**。
- **形状**：输入每 rank 一份完整 `data`，输出每 rank 一份 `data/N`。
- **通信量**：约 `data × (N-1)/N`（All-Reduce 的一半，因为不需要把完整结果再发一遍）。
- **典型场景**：**FSDP2 的反向传播**——每个 rank 算出完整梯度后，reduce-scatter 把"归约求和后的梯度"按参数切片分发，每个 rank 只保留自己分片对应的那部分梯度（配合优化器状态分片）。

### 2.4 All-Gather（全局收集）

- **语义**：每个 rank 持有 `data` 的一个分片（`data/N`），操作后**每个 rank 都拿到拼接后的完整 `data`**。
- **形状**：输入每 rank 一份 `data/N`，输出每 rank 一份完整 `data`。
- **通信量**：约 `data × (N-1)/N`。
- **典型场景**：**FSDP2 的前向传播**——每个 rank 只存了参数的一个分片，forward 前先 all-gather 出完整参数用于计算，用完再释放（`reshard_after_forward`）。

**All-Reduce = Reduce-Scatter + All-Gather**：这是理解 FSDP 和 DDP 关系的关键——FSDP2 把 DDP 里"一次 all-reduce"拆成了"反向 reduce-scatter + 前向 all-gather"两步，中间还顺便把参数/梯度/优化器状态都分片存储了。**用通信次数换显存**，是 FSDP 的核心权衡。

```
rank0:[d0]  rank1:[d1]  rank2:[d2]  rank3:[d3]
        \-------- all-gather --------/
rank0:[d0,d1,d2,d3]  rank1:[同上]  rank2:[同上]  rank3:[同上]
```

### 2.5 All-to-All（全交换）

- **语义**：每个 rank 给**每一个其他 rank**都准备了一份**不同**的数据（可以理解为一个 `N×N` 的数据矩阵，第 `i` 行是 rank `i` 发出的数据，第 `j` 列是 rank `j` 收到的数据），操作后每个 rank 拿到"所有 rank 发给自己的那一份"拼在一起。
- **形状**：跟前面几个不同，**每个 rank 发出/收到的数据量可能不相等**（尤其是 MoE 场景，token 路由不均衡时）。
- **通信量**：最坏情况下是 `O(N²)` 条消息（每对 rank 之间都要发），是几个原语里通信模式最复杂、最容易产生负载不均衡的一个。
- **典型场景**：**EP（专家并行）的 token dispatch**——每个 rank 上的 token 被路由到不同专家、不同专家在不同 rank 上，所以要把 token "发给正确的 rank"，算完再 all-to-all 一次把结果发回原 rank。**CP（Ring Attention）**里 K/V 的环形交换本质上也是一种结构化的点对点交换（不是标准 all-to-all，但通信模式类似，逐步交换而非一次性全交换）。

```
rank0: [->r0, ->r1, ->r2, ->r3]
rank1: [->r0, ->r1, ->r2, ->r3]
rank2: [->r0, ->r1, ->r2, ->r3]
rank3: [->r0, ->r1, ->r2, ->r3]
        \-- 每个 rank 只收集"发给自己"的那一列 --/
```

### 2.6 点对点 Send/Recv（非 collective，但同样重要）

- **语义**：只在两个特定 rank 之间传数据，不涉及其他 rank。
- **典型场景**：**PP（流水线并行）**——stage `i` 算完的激活值只需要发给 stage `i+1`，反向的梯度只需要发给 stage `i-1`，不需要全局同步。这也是为什么 PP 通信量远小于 TP/DP，适合放在跨机（高延迟、低带宽）维度。

### 2.7 小结：并行方式 <-> 通信原语对照表

| 并行方式 | 主要用到的通信原语 | 通信频率/量级特点 |
|---|---|---|
| DDP | All-Reduce | 每次反向传播一次，量 = 全部梯度 |
| FSDP2 | All-Gather（前向）+ Reduce-Scatter（反向） | 每层/每组参数一次，比 DDP 更细粒度 |
| TP | All-Reduce / All-Gather（配合 DTensor 自动插入） | 每个 TP 切分层一次，量小但频率高，要求高带宽 |
| PP | 点对点 Send/Recv | 只在 stage 边界，量 = 激活值/梯度，不是全局 |
| CP | 环形 Send/Recv（Ring Attention） | 逐步交换 K/V，量随 CP 度增长 |
| EP | All-to-All | token dispatch + combine 两次，量可能不均衡 |

带着这张表去看后面每一周的 profiler trace，会比死记"TP 用 all-reduce"这种结论有用得多——**关键是看懂"为什么这个并行方式必须用这个原语"**（数据依赖关系决定了通信模式）。

---

## 3. 并行方式的分类：模型无关 vs 模型内部

这是本周要建立的第二个重要认知框架。

### 3.1 模型无关的并行（Model-Agnostic）—— DP 系列

**DP/DDP/FSDP2/HSDP 完全不需要理解模型内部结构**：复制（或分片）整个模型，把**数据**切分到不同 rank，每个 rank 独立跑完整的前向+反向，最后同步梯度。

- 优点：**通用性极强**——任何模型不用改代码就能套上 DDP/FSDP。这也是为什么 torchtitan 里 `deepseek_v4/sharding.py` 应用 FSDP 的部分相对"标准化"，不需要针对 DeepSeek V4 的稀疏 Attention/MoE 结构写特殊逻辑（专家参数的 FSDP 分片是个例外，见第 1 周内容里提到的 `efsdp`）。
- 局限：**没有解决单卡放不下"一份完整模型/一份完整前向"的问题**——DDP 每张卡还是要放完整模型；FSDP2 虽然把参数/梯度/优化器状态分片了，但**计算时依然要 all-gather 出完整参数**，且**激活值不切分**（一条样本的前向依然要在一张卡上完整跑完）。

### 3.2 模型内部的并行（Model-Internal）—— TP / PP / CP / EP

这四种都需要**切开模型计算图本身**，因此需要理解模型结构才能正确切分：

- **TP**：切开单层内部的矩阵乘法（比如 Attention 的 QKV 投影按 head 切、MLP 按 hidden_dim 切）。**需要知道具体是哪个 Linear、维度语义是什么**——这也是为什么 `deepseek_v4/sharding.py` 里专门为稀疏 Attention（`set_deepseek_v4_attention_sharding`）、压缩器（`set_compressor_sharding`）、Indexer（`set_indexer_sharding`）写了单独的分片规则：标准 Attention 的 TP 切分规则套不到这种结构上。
- **PP**：切开层与层之间的顺序执行（哪几层放 stage 0，哪几层放 stage 1）。**需要模型是"层可分割"的**——DeepSeek V4 的 MTP（Multi-Token Prediction）分支就是一个不能简单当成"纯 for 循环 decoder"来切 PP 的例子（第 3 周会展开）。
- **CP**：切开一条序列内部的 token 维度（哪些 token 放哪个 rank），需要重写 Attention 的计算方式（Ring Attention）,因为 Attention 本质上需要"看到全部 token"才能算完整的 attention。
- **EP**：切开 MoE 层内部不同专家的位置（哪个专家放哪个 rank），需要模型本身就是 MoE 结构才有意义。

**为什么这个分类很重要**：它直接决定了"改一个新模型接入某种并行方式的工程成本"——接入 DP/FSDP 几乎零成本（模型代码不用改），接入 TP/PP/CP/EP 都需要针对模型结构写专门的切分规则。torchtitan 的设计哲学（也是第 6 周要精读的 `sharding.py`）就是尽量把"模型内部并行"的规则也做成可配置、可复用的声明式代码，而不是散落在模型 forward 里写 if-else。

---

## 4. 动手：跑通脚本 + 两个"坑"背后的原理

### 4.1 先跑起来

```bash
./scripts/run_deepseek_debug.sh
```

脚本只有 60 行（见 [`scripts/run_deepseek_debug.sh`](../scripts/run_deepseek_debug.sh)），核心逻辑是把 `MODULE=deepseek_v4 CONFIG=deepseek_v4_debugmodel` 固定后转发给 `run_train.sh`，同时做了两处覆盖来让它能在单卡消费级 GPU 上跑：`SEQ_LEN`（默认 4096）和 `CUDA_GRAPHS`（默认 0）。

### 4.2 坑 1：O(n²) 稠密 mask —— 显存墙的具体案例

已经在第 1.1 节详细算过：`deepseek_v4/attention.py::_build_block_mask` 里的 `selected_count` 缓冲区是 `[1, seqlen, kv_len]` 的稠密 int32 张量，**大小随序列长度平方增长**：

- `seqlen=16384`（debugmodel 默认，还会被 8 倍折叠成 131072 的 stream）→ 单层 ~64 GiB，直接打爆任何单卡。
- `seqlen=4096`（脚本覆盖后）→ 单层 ~64 MiB，可控。

**这是一个具体、可复现的"显存墙"案例**：不是"模型太大"，而是**某个中间数据结构的显存增长率是平方级**，这种情况普通的"参数量估算公式"完全捕捉不到，必须读代码才能发现。这也是为什么第 4 周的 CP（Context Parallel）专门把这个 mask 拿出来作为切入点——CP 的价值之一就是把这个 O(n²) 的峰值显存摊到多个 rank 上。

**动手练习**：把 `SEQ_LEN` 从 4096 逐步调大（8192、16384），配合 `watch nvidia-smi` 观察显存曲线是不是按平方增长（而不是线性）。可以先估算一个理论显存需求，再跟实测对比误差。

### 4.3 坑 2：CUDA Graph + MoE dispatcher 的非法拷贝 —— 通信墙/工程约束的具体案例

脚本注释解释:CUDA Graph 捕获期间不允许有 CPU<->GPU 同步拷贝，而标准 MoE token dispatcher（`LocalTokenDispatcher`，`expert_parallel_degree=1` 时使用）内部在 `torch._grouped_mm` 里有一次这样的拷贝，`Trainer._validate_cuda_graphs` 目前又没有检测出这个组合是非法的,所以脚本默认关掉 CUDA Graph（`CUDA_GRAPHS=0`）来避免直接报错或产生错误结果。

这提前预告了第 5 周要处理的问题：**EP≥2 时会换成 graph-safe 的 dispatcher，理论上可以重新打开 CUDA Graph**。这是一个很好的例子说明"看似独立的两个特性（CUDA Graph 加速 vs MoE 并行方式）之间可能存在隐藏的耦合约束"——这种约束在读文档时很难发现，只有跟着报错信息或者代码里的注释一层层刨才能搞清楚。

### 4.4 延伸练习：GPU-Util ≠ 显存占用

跑的时候如果 `watch nvidia-smi`，会看到两个容易混淆的指标：

- **Memory-Usage**（如 `2360MiB / 16311MiB`）：真实显存占用，对应第 1.1 节的分项估算。
- **GPU-Util**（如 `95%`）：这是**过去采样周期内 GPU 上"有没有 kernel 在跑"的时间占比**，跟显存无关，也不等于"算力被用满了多少"。

对这种 `dim=256, n_layers=4` 的极小 debug 模型，GPU-Util 跑到 95% 很正常，但原因往往不是"计算密集"，而是**大量小 kernel（indexer/compressor/block mask 构建/MoE 路由分发/多层小 GEMM）连续排队执行**，GPU 调度器几乎没有空闲窗口，属于 **launch-overhead-bound**，不是 **compute-bound**。真正的算力利用率（SM efficiency / achieved occupancy）需要用 `torch.profiler` 或 Nsight 才能看到,这也是第 6 周"性能分析"里 MFU 指标存在的意义——MFU 才是衡量"算力有没有被真正用满"的指标,GPU-Util 不是。

**动手练习**：用 `torch.profiler` 抓一次 trace（可以参考项目自带的 `cuda_graph_trace_compaction` 技能处理带 CUDA Graph 的 trace），确认这个猜测——看 trace 里是大量几十微秒的小 kernel 排队，还是少数大 GEMM 占主导。

---

## 5. 推荐阅读（带阅读重点）

- **torchtitan 论文 [§2 Background](https://arxiv.org/abs/2410.06511)**：重点看它怎么把"三堵墙"和"五种并行原语"对应起来讲的，建立跟本文一致的心智模型。
- **PyTorch 官方 [Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)**：重点看 collective 通信原语的图示，跟第 2 节的表格对照着看。

---

## 6. 自测清单

学完这一周,应该能不看这篇文档回答：

1. 显存墙的四个组成部分是什么？哪个是"静态"的，哪个是"动态、跟序列长度/batch 相关"的？
2. All-Reduce、Reduce-Scatter、All-Gather 三者的输入输出形状分别是什么？三者的数学关系是什么？
3. 为什么 TP 几乎只在机内用，PP 可以跨机用？（提示：通信量 vs 通信模式的差异）
4. "模型无关的并行"和"模型内部的并行"分类标准是什么？为什么这个分类决定了接入新模型的工程成本？
5. `deepseek_v4_debugmodel` 里为什么 `SEQ_LEN` 从 16384 降到 4096 能让显存从 ~64GiB 降到 ~64MiB？（说出具体是哪个张量、什么增长规律）
6. `nvidia-smi` 里的 `GPU-Util 95%` 能不能直接当成"显存快用满了"或"算力被用满了"的证据？为什么？
7. `6ND` 里的 `6` 是怎么来的？为什么反向传播的 FLOPs 是前向的 2 倍？这个公式在什么情况下会明显低估实际算力需求？

如果这几个问题都能讲清楚（哪怕讲给自己听），第 0 周的目标就达成了，可以进入 [第 1 周：数据并行](./distributed-training-roadmap.md#第-1-周数据并行--ddp--fsdp2--hsdp)。
