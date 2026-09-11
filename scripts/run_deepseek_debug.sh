#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Single-GPU smoke test for DeepSeek V4 debugmodel.
# Wraps run_train.sh with MODULE/CONFIG pinned to deepseek_v4/debugmodel.
# Override via env vars, e.g.:
#    STEPS=5 OUTPUT=./outputs/deepseek_debug_run2 ./scripts/run_deepseek_debug.sh
# Any extra tyro overrides are still forwarded, e.g.:
#    ./scripts/run_deepseek_debug.sh --training.steps 5

cd "$(dirname "${BASH_SOURCE[0]}")/.."

NGPU=${NGPU:-1}
STEPS=${STEPS:-50}
OUTPUT=${OUTPUT:-"./outputs/deepseek_debug"}

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

NGPU="${NGPU}" \
MODULE="deepseek_v4" \
CONFIG="deepseek_v4_debugmodel" \
./run_train.sh \
    --training.steps "${STEPS}" \
    "${EXTRA_ARGS[@]}" \
    --dump_folder "${OUTPUT}" \
    "$@"
