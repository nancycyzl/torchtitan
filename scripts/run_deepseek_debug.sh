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
STEPS=${STEPS:-10}
OUTPUT=${OUTPUT:-"./outputs/deepseek_debug"}

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
    --dump_folder "${OUTPUT}" \
    "$@"
