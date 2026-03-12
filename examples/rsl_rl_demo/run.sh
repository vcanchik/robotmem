#!/bin/bash
# rsl_rl 适配器 Docker 验证 — 一键构建运行
# 用法: bash examples/rsl_rl_demo/run.sh
set -e
cd "$(dirname "$0")/../.."
echo "=== 构建 Docker 镜像（torch CPU + rsl_rl + robotmem）==="
docker build -t robotmem-rsl-rl -f examples/rsl_rl_demo/Dockerfile .
echo ""
echo "=== 运行验证 ==="
docker run --rm robotmem-rsl-rl
