# 第 1 周：数据并行 —— DDP → FSDP2 → HSDP

> 本文是 [`distributed-training-roadmap.md`](./distributed-training-roadmap.md) 第 1 周内容的展开讲解。
> 前置知识：[第 0 周：分布式训练基础](./week0-distributed-fundamentals.md)（三堵墙、collective 通信原语、显存估算）。
> 目标：搞懂 DDP、ZeRO、FSDP2、HSDP 之间"一脉相承"的关系——它们都在回答同一个问题："显存墙"里的参数/梯度/优化器状态，到底该不该切、怎么切。
> 示例模型统一用仓库自带的 `deepseek_v4_debugmodel`，配合单卡脚本 [`scripts/run_deepseek_debug.sh`](../scripts/run_deepseek_debug.sh)。

---

## 1. DDP：先把"模型无关的并行"讲透

回顾第 0 周 3.1 节的分类：DDP 是**模型无关**的并行——不需要理解模型内部结构，只需要"复制模型，切分数据"。

**DDP 的运行方式：**
1. 每张卡（rank）都持有一份**完整**的模型参数副本。
2. 每张卡从全局 batch 里切一份不重叠的子 batch，独立跑完整的前向 + 反向。
3. 反向传播算出的**梯度**互不相同（因为各卡数据不同），需要做一次 **All-Reduce**（第 0 周 2.2 节）把梯度求和/求平均，这样每张卡拿到的是全局梯度，用它更新参数，保证下一步所有卡的参数还是一致的。

**DDP 没有解决什么**：显存墙的四个组成部分（参数/梯度/优化器状态/激活值）里，DDP **一个都没切**——只是把"用多少张卡处理多少数据"这件事并行化了。每张卡依然要放得下完整的模型参数 + 梯度 + 优化器状态 + 一条子 batch 的激活值。这正是第 0 周显存公式里算出的 "170 MiB" 那部分——DDP 下这 170 MiB 在**每张卡上都存一份**，卡数越多，这份冗余存储的总浪费越大。

**DDP 的定位**：适合"模型本身单卡放得下，只是想加快训练速度/喂更多数据"的场景。一旦模型大到单卡放不下（这是大模型训练的常态），就需要下面的 ZeRO/FSDP。

---

## 2. ZeRO 思想：用通信换显存

DeepSpeed 的 ZeRO（Zero Redundancy Optimizer）论文提出的核心洞察：DDP 里"每张卡都存一份完整的参数+梯度+优化器状态"是**纯冗余**——`N` 张卡其实只需要合起来存一份就够，只不过要按 `N` 份切开，每张卡存 `1/N`。需要完整数据的时候（比如某一层要做前向计算），再临时用通信把它拼回来。

ZeRO 分三个阶段，逐步扩大"切分范围"：

| 阶段 | 切分对象 | 未切分对象 |
|---|---|---|
| ZeRO-1 | 优化器状态（Adam 的 m, v） | 参数、梯度仍完整存 |
| ZeRO-2 | 优化器状态 + 梯度 | 参数仍完整存 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 无——最激进，显存收益最大，通信也最多 |

**代价是通信**：某一层要计算前向时，如果参数是切分存储的，就必须先 All-Gather 出完整参数（第 0 周 2.4 节），算完再释放；反向传播算出的梯度也不再是本地就能用的完整梯度，需要 Reduce-Scatter（第 0 周 2.3 节）分发到对应分片。**ZeRO-3 约等于把 DDP 的"一次 All-Reduce"拆成了"前向 All-Gather + 反向 Reduce-Scatter"两次通信**——这正是第 0 周 2.4 节"All-Reduce = Reduce-Scatter + All-Gather"这个等式在工程上的落地。

PyTorch 的 **FSDP（Fully Sharded Data Parallel）** 就是 ZeRO-3 思想的原生实现。

---

## 3. FSDP2：torchtitan 的默认数据并行方案

### 3.1 为什么是 FSDP2，不是 FSDP1

torchtitan 用的是 **FSDP2**（[`docs/fsdp.md`](../docs/fsdp.md)），是 PyTorch 对原版 FSDP（这里称 FSDP1）的重写。核心区别：

- **FSDP1** 把一组参数打平拼接成一个 `FlatParameter`（一个大 1D 张量）来做通信分桶，但这让"对单个参数做不同处理"（冻结、单独转精度……）变得很别扭，`state_dict` 逻辑也因此变得极其复杂。
- **FSDP2** 去掉了 `FlatParameter`，改用 **`DTensor`**（PyTorch 原生分布式张量抽象）表示每个分片参数，`Shard(0)`（沿 dim-0 切分）是默认的分片 placement。`model.named_parameters()` 拿到的参数名不变，只是每个参数的类型变成了 `DTensor`。
- FSDP2 用 `fully_shard(module)` 这个函数式 API，而不是像 FSDP1 那样用 `nn.Module` 包一层——它是通过 `@contract` 装饰器给 `module` 挂一个 `FSDPState`，再做一次"动态换基类"（比如 `Transformer` 被换成 `FSDPTransformer(FSDPModule, Transformer)`），这样可以在不改变模块层级结构的前提下加新方法。

### 3.2 `fully_shard` 在做什么

关键参数（对照 [`docs/fsdp.md`](../docs/fsdp.md) 的 API 表）：

```python
fully_shard(
    module,
    mesh=None,                              # 1D mesh 用于 FSDP，2D mesh 用于 HSDP
    reshard_after_forward=True,             # True=ZeRO-3, False=ZeRO-2
    mp_policy=MixedPrecisionPolicy(),       # 混合精度策略
    offload_policy=OffloadPolicy(),         # CPU offload
)
```

`reshard_after_forward` 是最值得记住的一个开关，直接对应 ZeRO 的阶段选择：

| `reshard_after_forward` | 行为 | 对应 ZeRO/DeepSpeed 阶段 |
|---|---|---|
| `True` | 前向算完就释放（reshard）参数分片，反向时再重新 All-Gather 一次 | ZeRO-3 |
| `False` | 前向算完不释放，参数留在内存里到反向直接用，省一次 All-Gather | ZeRO-2（`SHARD_GRAD_OP`） |
| `int`（如 8）| 前向后 reshard 到一个更小的 world size（比如 8，可以理解成"机内"），反向的 All-Gather 只在这个小组内做 | ZeRO++ 的 hpZ |

**这是一个纯粹的"显存 vs 通信"权衡**：`True` 显存最省但通信最多（每层前向反向各一次 All-Gather），`False` 省一次通信但要多存一份未释放的参数分片。

### 3.3 torchtitan 怎么把 FSDP2 接到模型上：`apply_fsdp_to_decoder`

通用逻辑在 [`torchtitan/distributed/fsdp.py`](../torchtitan/distributed/fsdp.py) 的 `apply_fsdp_to_decoder()`（`llama3`/`qwen3`/`deepseek_v3`/`gpt_oss` 等模型的 `parallelize.py` 都直接调它；DeepSeek V4 走的是另一套声明式路径，见第 5 节）。几个值得精读的设计点：

1. **不是整个模型一次 `fully_shard`，而是按"通信单元"分层包**：
   - `tok_embeddings` 单独一个 FSDP unit；
   - 权重绑定（`enable_weight_tying`）时，`tok_embeddings` + `norm` + `lm_head` 打包成**一个** unit（避免重复 All-Gather 同一份共享权重）；
   - 每个 `transformer_block` 单独一个 unit；
   - 最外层再对整个 `model` 包一次 `fully_shard`。
   
   这样切的原因：**FSDP 的通信粒度 = `fully_shard` 包的粒度**。包得太粗（比如整个模型一个 unit）就没法把"某一层的计算"和"下一层参数的 All-Gather"重叠起来；包得太细则通信次数暴增、小消息多、延迟占比上升。按 `transformer_block` 切正好在这两者之间取得平衡——这也是为什么第 0 周 2.7 节表格里说 FSDP2 通信"比 DDP 更细粒度"。

2. **"最后几层不 reshard" 的小优化**（`fsdp.py:258-265`）：`norm` + `lm_head` 默认不用 `reshard_after_forward` 策略里算出的值，而是直接看 policy 是不是 `"always"`。原因是这两个模块在前向里是**最后**被用到的，FSDP 的 prefetch 机制会在它们前向执行前就把参数 All-Gather 好，如果 reshard 掉马上又要为反向重新 gather 一次，纯属浪费。

3. **`disable_fsdp_gradient_division`**：torchtitan 自己在训练循环里用全局 token 数做梯度缩放，所以关掉 FSDP 内置的梯度平均（`set_gradient_divide_factor(1.0)`），避免被除两次。这是一个"两套机制做同一件事，必须关掉一个"的典型例子。

### 3.4 FSDP2 的显存/通信时间线（结合第 0 周原语）

一个 FSDP transformer block 的完整生命周期：

```
前向：All-Gather 本层参数分片 -> 计算前向 -> (reshard_after_forward=True 时)释放完整参数，只留分片
反向：All-Gather 本层参数分片(若上一步释放了) -> 计算反向，得到完整梯度 -> Reduce-Scatter 梯度 -> 每个 rank 只保留自己分片对应的梯度
优化器 step：每个 rank 只更新自己那部分参数分片对应的优化器状态
```

用 [`docs/debugging.md`](../docs/debugging.md) 的 profiling 工具抓一次 trace（若有多卡环境），应该能在 timeline 上看到：每个 transformer block 前面有一小段 All-Gather，反向每个 block 后面有一小段 Reduce-Scatter，且理想情况下这些通信应该和**相邻层的计算**重叠（这就是 FSDP 的 prefetch 机制要做的事）。

---

## 4. HSDP：`dp_replicate x dp_shard` 二维组合

HSDP（Hybrid Sharded Data Parallel）本质是"组内 FSDP + 组间 DDP"：把 `N` 张卡分成 `dp_replicate` 组，每组内部 `dp_shard` 张卡做 FSDP 分片，组之间做 DDP 式的梯度 All-Reduce（严格说是组间也走 reduce-scatter/all-gather 的 DTensor 路径，但语义等价于"组间复制、组内分片"）。

**为什么需要 HSDP，仅靠更大的 FSDP 度不行吗？**

1. **跨机通信成本**：纯 FSDP 的 `dp_shard` 度如果跨机器铺开，每一层的 All-Gather/Reduce-Scatter 都要走机间网络（带宽远低于机内 NVLink，参考第 0 周 1.3 节通信墙）。HSDP 把 `dp_shard` 限制在机内（NVLink 域），`dp_replicate` 放到跨机维度——机间只需要做梯度同步，通信量和频率都比"跨机 FSDP"小得多。
2. **容错/重启成本**：`dp_replicate` 组之间是完整的参数副本关系，某些容错训练方案（如 torchtitan 第 7 周会碰到的 TorchFT）可以利用这种冗余做更细粒度的故障恢复，而不需要重启整个训练。

在 torchtitan 里，`mesh` 参数在 `fully_shard` 里从 1D 变成 2D 就自动切换到了 HSDP 语义（[`docs/fsdp.md`](../docs/fsdp.md) 里 `mesh` 那条：1D mesh 用于 FSDP，2D mesh 用于 HSDP，约定第 0 维是 replicate、第 1 维是 shard）。[`torchtitan/distributed/fsdp.py`](../torchtitan/distributed/fsdp.py) `apply_fsdp_to_decoder` 末尾有一处判断：

```python
if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
    logger.info("Applied HSDP to the model")
else:
    logger.info("Applied FSDP to the model")
```

也就是说 **HSDP 不是一套独立实现，只是 FSDP2 mesh 多了一维**——这跟第 0 周强调的"torchtitan 用统一 `DeviceMesh` 组织并行"的设计是一致的。第 6 节会用这条日志实际验证一次。

**一个容易漏掉的点（第 6 节会用实验验证）**：HSDP 下单个 rank 的参数/梯度/优化器状态显存，只取决于 `dp_shard` 这一维的宽度，跟 `dp_replicate` 是多少无关——因为"分片"只发生在 `dp_shard` 维度上，`dp_replicate` 维度上每组都是完整（但各自分片好的）一份。

---

## 5. DeepSeek V4 的特殊之处：`spmd_types` 声明式后端 + `efsdp`

precise 一点说：`llama3`/`qwen3`/`deepseek_v3`/`gpt_oss` 等模型的 `parallelize.py` 是**命令式**地直接调用 `apply_fsdp_to_decoder()`；而 **DeepSeek V4 走的是另一条路**——[`torchtitan/models/deepseek_v4/sharding.py`](../torchtitan/models/deepseek_v4/sharding.py) 用 `ShardingConfig`/`spmd_types` 声明每个张量在各个 mesh 轴上应该是什么 placement（`spmd.R` 复制、`spmd.S(dim)` 分片……），具体怎么落地成 `fully_shard` 调用则交给通用逻辑处理。这也是第 6 周要精读的 `--parallelism.spmd_backend spmd_types` 选项的来源，本周只需要知道"存在这两条路"，不用深究声明式系统内部机制。

跟本周直接相关的，是 [`torchtitan/distributed/fsdp.py`](../torchtitan/distributed/fsdp.py) 里两个专门给 `spmd_types` 用的函数：

```python
_DENSE_STORAGE_AXES = ["dp_replicate", "dp_shard", "cp", "tp"]
_SPARSE_STORAGE_AXES = ["dp_replicate", "efsdp", "ep"]

def resolve_fsdp_mesh(parallel_dims): ...        # 稠密参数（非专家）的 FSDP mesh
def resolve_sparse_fsdp_mesh(parallel_dims): ...  # 专家参数专用的 FSDP mesh（efsdp 轴）
```

- **非专家参数**（Attention、共享 FFN、norm……）走 `dp_replicate`/`dp_shard`/`cp`/`tp` 这几个轴，和普通模型没有本质区别，本周内容直接适用。
- **路由专家（routed experts）参数**走的是**另一个独立的轴 `efsdp`**（`resolve_sparse_fsdp_mesh` 只在 `parallel_dims.ep_enabled` 时才返回非空），`dp_replicate` 在两套 mesh 间共享，但 FSDP 分片的宽度是 `efsdp`，不是 `dp_shard`——这是因为专家参数还要再叠一层专家并行（EP，第 5 周），`efsdp` 和 `ep` 两个轴会先合并算出专家参数总共分布在多少个 rank 上，再决定怎么切：

```python
# torchtitan/distributed/fsdp.py apply_fsdp_to_decoder 内
if ep_degree > 1:
    efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
else:
    efsdp_ep_size = fsdp_config["mesh"].size()

if efsdp_ep_size > num_experts:
    expert_shard_placement = Shard(1)   # 专家数不够分，退化成按隐藏维切，避免 padding
else:
    expert_shard_placement = Shard(0)   # 默认按"专家"这一维切（每个 rank 分到完整的若干个专家）
```

也就是说：默认情况下专家参数是按**专家数量**这一维（`Shard(0)`，对应张量形状里的 `E` 维，参考 `deepseek_v4/sharding.py` 里 `_GROUPED_EXPERTS_PARAM_LAYOUT = {"w1_EFD": spmd.S(1), ...}` 这种 `w1_EFD` 命名——`E` 是专家数维）分片；只有当参与分片的 rank 数**超过专家数**时（比如 8 个专家但 `efsdp_ep_size=16`），才退化成按 FFN 隐藏维（`F`/`D`）分片，否则会有 rank 分不到完整专家、产生 padding 浪费。

**本周先记住这个结论，具体 EP 怎么和 FSDP 交互，第 5 周会展开**：这里的关键认知是——**专家参数的 FSDP 分片规则和普通稠密参数不是一套逻辑**，`efsdp` 是专门为此单独开的一个 mesh 轴，这也是第 0 周表格里为什么专门把 `ep`/`efsdp` 列成一个独立的轴，而不是把专家参数直接塞进 `dp_shard`。

---

## 6. 动手

### 6.1 这个环境的限制，以及怎么绕过去

当前开发机是**单卡**（`RTX 5060 Ti, 16GB`）。roadmap 里 `NGPU=4` 的例子在这台机器上**跑不起来**（`torchrun --nproc_per_node=4` 会尝试用 4 个进程抢同一张物理 GPU 并各自走 NCCL，正常会失败或行为不可信）。如果你有多卡机器，直接跳到 6.3 节用真实命令；单卡环境下可以做的是：

**用 `COMM_MODE=fake_backend`（[`docs/debugging.md`](../docs/debugging.md) "Fake Backend Debugging" 一节）在单卡上验证任意 `NGPU` 的并行配置**——它用假的 process group 模拟通信（不做真实的跨卡数据搬运），只在单卡、单进程、不需要 `torchrun`/NCCL 初始化的情况下，把 mesh 构建、模型分片、rank-0 的训练循环逻辑完整跑一遍。**局限**：不能用来测真实吞吐/延迟（通信是假的，测不出真实开销），也不能验证依赖真实数据交换正确性的逻辑；但验证"mesh 建对了没"、"FSDP/HSDP 有没有生效"、"显存分片有没有生效"完全够用。

### 6.2 实测：`dp_shard` 度如何影响单 rank 显存

```bash
for shard in 1 2 4 8; do
  NGPU=$shard COMM_MODE=fake_backend MODULE=deepseek_v4 CONFIG=deepseek_v4_debugmodel ./run_train.sh \
    --parallelism.data_parallel_shard_degree $shard \
    --training.num_tokens_per_microbatch_per_dp_rank 4096 \
    --training.max_context_length 4096 \
    --training.disable_cuda_graphs
done
```

**实测结果**（本机，`deepseek_v4_debugmodel`，10,601,565 参数）：

| `dp_shard` | 日志 "Building device mesh" | "CUDA memory usage for model"（分片后的参数+优化器状态显存） |
|---|---|---|
| 1（无 FSDP，等价 DDP） | `dp_shard=1` | 0.08 GiB (0.53%) |
| 2 | `dp_shard=2` | 0.04 GiB (0.29%) |
| 4 | `dp_shard=4` | 0.03 GiB (0.21%) |
| 8 | `dp_shard=8` | 0.03 GiB (0.18%) |

**怎么解读**：随 `dp_shard` 翻倍，这部分显存大致减半（1 -> 2 减半明显，之后因为这个 debug 模型只有 10.6M 参数、GiB 显示只保留两位小数，4/8 之间的差异已经小到被舍入抹平，但百分比列仍然看得出持续下降趋势）。这正是 FSDP "参数按 rank 数切分存储" 最直接的证据——**在真实规模的模型上（十亿参数级别），这个降幅会是显存曲线上非常显著的一段，不会像这里一样被固定开销盖住**（回忆第 0 周结论："固定开销远大于模型本身"只在 debug 模型这种极小规模下成立）。

**注意**：`memory: 1.4x GiB` 那一行（step 打印里的总显存）几乎不随 `dp_shard`变化——这是因为这台单卡机器在 fake backend 下所有"rank"共享同一张物理卡，这个总显存里绝大部分是 CUDA context/allocator 的固定开销（第 0 周 1.1 节），不能用来判断 FSDP 效果，**要看的是 "CUDA memory usage for model" 这一行**，它是分片后参数本身的显存占用。

### 6.3 实测：HSDP 的 mesh 构建 + "Applied HSDP" 日志

```bash
NGPU=4 COMM_MODE=fake_backend MODULE=deepseek_v4 CONFIG=deepseek_v4_debugmodel ./run_train.sh \
  --parallelism.data_parallel_replicate_degree 2 \
  --parallelism.data_parallel_shard_degree 2 \
  --training.num_tokens_per_microbatch_per_dp_rank 4096 \
  --training.max_context_length 4096 \
  --training.disable_cuda_graphs
```

**实测日志**（对照第 3.3/4 节的代码）：

```
Building device mesh with parallelism: pp=1, dp_replicate=2, dp_shard=2, cp=1, tp=1, ep=1
Successfully created meshes with active dimensions: ['batch', 'loss', 'dp_replicate', 'dp', 'dp_shard']
Applied HSDP to the model
CUDA memory usage for model: 0.04GiB(0.29%)
```

跟第 4 节的结论对上了两件事：
1. `"Applied HSDP to the model"` 这条日志确实来自 `fsdp.py` 里 `"dp_replicate" in dp_mesh.mesh_dim_names` 的判断——mesh 名字列表里多了 `dp_replicate` 这一维。
2. **`CUDA memory usage for model` 是 `0.04GiB`，跟 6.2 节里纯 FSDP `dp_shard=2` 的结果完全一致**——印证了第 4 节的结论："单 rank 显存只取决于 `dp_shard` 宽度，跟 `dp_replicate` 无关"。这里 `dp_replicate=2 x dp_shard=2` 和纯 `dp_shard=2`（`dp_replicate=1`）相比，总卡数翻倍了，但单卡显存分片粒度不变——多出来的卡是拿去做"组间复制换吞吐/容错"，不是拿去"切更细的显存分片"。

### 6.4 如果你有真实多卡环境

```bash
# 纯 FSDP（dp_shard=4）
NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.data_parallel_shard_degree 4

# HSDP：2 组复制 x 每组 2 卡分片
NGPU=4 ./scripts/run_deepseek_debug.sh \
  --parallelism.data_parallel_replicate_degree 2 \
  --parallelism.data_parallel_shard_degree 2
```

用 [`docs/debugging.md`](../docs/debugging.md) 里的 profiling 工具（`--profiler.enable_memory_snapshot` 做显存 timeline，或 `torch.profiler` trace）抓一次真实 trace，重点看两件事：
1. **通信 timeline**：All-Gather（前向）/ Reduce-Scatter（反向）是否出现在预期位置（每个 transformer block 前后），以及是否跟相邻层计算**重叠**（理想情况下通信 kernel 应该和计算 kernel 在时间轴上并行，而不是先等通信完再算）。
2. **显存曲线**：跟 6.2 节单卡 fake_backend 测到的相对趋势做对比——在真实规模模型上，`dp_shard` 翻倍带来的显存降幅应该比这里明显得多。

---

## 7. 推荐阅读（带阅读重点）

- **[`docs/fsdp.md`](../docs/fsdp.md)**：重点看 `reshard_after_forward` 和 ZeRO/DeepSpeed 阶段的对照表（3.2 节已经摘录），以及 Meta-Device Initialization 一节（`with torch.device("meta")` 配合 `fully_shard` 再 `to_empty`，理解为什么 FSDP2 不再需要 `param_init_fn`）。
- **[`torchtitan/distributed/fsdp.py`](../torchtitan/distributed/fsdp.py)**：整个文件不到 450 行，通读一遍，重点是 `apply_fsdp_to_decoder` 的分层包裹逻辑和 `resolve_fsdp_mesh`/`resolve_sparse_fsdp_mesh` 这两个 `spmd_types` 专用函数。
- **[`torchtitan/models/deepseek_v4/sharding.py`](../torchtitan/models/deepseek_v4/sharding.py)**：不用通读，搜索 `_GROUPED_EXPERTS_PARAM_LAYOUT` 看专家参数的声明式 placement 长什么样即可，呼应第 5 节。
- **[`docs/debugging.md`](../docs/debugging.md)** "Fake Backend Debugging" 一节：本周动手部分用到的核心工具，建议完整读一遍，了解它的边界（不能测性能、不能测数据依赖逻辑）。

---

## 8. 自测清单

学完这一周，应该能不看这篇文档回答：

1. DDP 切分了显存墙的哪几部分？为什么说 DDP "没有解决显存墙"？
2. ZeRO-1/2/3 分别切分什么？FSDP2 对应哪一个（默认情况）？
3. `reshard_after_forward=True/False/int` 分别对应什么行为、什么显存/通信权衡？分别对应哪个 DeepSpeed 概念？
4. `apply_fsdp_to_decoder` 为什么要按 `transformer_block` 这个粒度分别调用 `fully_shard`，而不是对整个模型只调用一次？如果只调用一次会有什么后果？
5. HSDP 里，单个 rank 的参数显存占用取决于 `dp_replicate` 还是 `dp_shard`？为什么？（提示：结合 6.3 节的实测数据）
6. DeepSeek V4 的路由专家参数为什么不能直接用 `dp_shard` 分片，而要单独开一个 `efsdp` 轴？`efsdp_ep_size > num_experts` 时分片策略会发生什么变化，为什么？
7. `COMM_MODE=fake_backend` 能验证什么、不能验证什么？为什么它能在单卡上"跑" `NGPU=128` 的配置？

如果这几个问题都能讲清楚，第 1 周的目标就达成了，可以进入 [第 2 周：张量并行 TP](./distributed-training-roadmap.md#第-2-周张量并行-tptensor-parallel)。
