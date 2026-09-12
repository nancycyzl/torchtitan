#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Smoke test for DeepSeek V4 debugmodel (defaults to 2 GPUs).
# Wraps run_train.sh with MODULE/CONFIG pinned to deepseek_v4/debugmodel.
# Override via env vars, e.g.:
#    STEPS=5 OUTPUT=./outputs/deepseek_debug_run2 ./scripts/run_deepseek_debug.sh
#    DP_SHARD=1 DP_REPLICATE=2 ./scripts/run_deepseek_debug.sh
# Any extra tyro overrides are still forwarded, e.g.:
#    ./scripts/run_deepseek_debug.sh --training.steps 5
#
# Every run's stdout/stderr is also teed into
# ${OUTPUT}/dsv4_debug_DPshard<N>_DPreplicate<M>_<timestamp>.log

cd "$(dirname "${BASH_SOURCE[0]}")/.."

NGPU=${NGPU:-1}
STEPS=${STEPS:-50}
OUTPUT=${OUTPUT:-"./outputs/deepseek_debug"}

# Data-parallel degrees, matching torchtitan's own defaults
# (data_parallel_shard_degree=-1 means "use all remaining ranks",
# data_parallel_replicate_degree=1 means "no HSDP replication").
DP_SHARD=${DP_SHARD:-1}
DP_REPLICATE=${DP_REPLICATE:-1}

# The deepseek_v4_debugmodel config defaults to a 16384 context folded x8 into a
# ~131072 token microbatch stream, and deepseek_v4/attention.py::_build_block_mask
# materializes a dense [1, stream_len, stream_len] mask (O(n^2), ~64GiB at the
# default). Shrink the per-microbatch token budget and context so the smoke test
# fits on a single commodity GPU. Override with SEQ_LEN=... for a larger run.
SEQ_LEN=${SEQ_LEN:-8192}

# CUDA graph capture is on by default, but the standard MoE token dispatcher
# (LocalTokenDispatcher, expert_parallel_degree=1) does a CPU<->CUDA copy inside
# torch._grouped_mm that is illegal during capture, and Trainer._validate_cuda_graphs
# does not catch it when expert_parallel_degree == 1. Disable graphs by default;
# set CUDA_GRAPHS=1 to opt back in (requires a graph-safe EP configuration).
CUDA_GRAPHS=${CUDA_GRAPHS:-0}

EXTRA_ARGS=(
    --parallelism.data_parallel_shard_degree "${DP_SHARD}"
    --parallelism.data_parallel_replicate_degree "${DP_REPLICATE}"
    --training.num_tokens_per_microbatch_per_dp_rank "${SEQ_LEN}"
    --training.max_context_length "${SEQ_LEN}"
)
if [ "${CUDA_GRAPHS}" = "0" ]; then
    EXTRA_ARGS+=(--training.disable_cuda_graphs)
fi

# Windows PyTorch builds are typically compiled without libuv support, which
# makes torchrun's default TCPStore backend fail at rendezvous. Fall back to
# the non-libuv TCPStore on Windows only; other platforms keep torch's default.
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) export USE_LIBUV=${USE_LIBUV:-0} ;;
esac

mkdir -p "${OUTPUT}"
LOG_FILE="${OUTPUT}/dsv4_debug_DPshard${DP_SHARD}_DPreplicate${DP_REPLICATE}_$(date +%Y%m%d_%H%M%S).log"

# Tee stdout+stderr (including the `set -x` trace) to a run-named log file
# while still printing live to the terminal.
exec > >(tee "${LOG_FILE}") 2>&1

# Once stdout is piped into tee (not a tty), CPython switches from
# line-buffered to fully block-buffered stdout, so torchrun's worker
# processes would otherwise hold log lines in memory until the buffer
# fills or the process exits. Force unbuffered I/O so the log file fills
# in as training progresses instead of all at once at the end.
export PYTHONUNBUFFERED=1

NGPU="${NGPU}" \
MODULE="deepseek_v4" \
CONFIG="deepseek_v4_debugmodel" \
./run_train.sh \
    --training.steps "${STEPS}" \
    "${EXTRA_ARGS[@]}" \
    --dump_folder "${OUTPUT}" \
    "$@"
