# robotmem rsl_rl 适配器 Docker 验证
# 用真实 torch + rsl_rl 跑 MemoryOnPolicyRunner，不用 mock
#
# 构建 & 运行:
#   docker build -t robotmem-rsl-rl -f examples/rsl_rl_demo/Dockerfile .
#   docker run --rm robotmem-rsl-rl

FROM python:3.11-slim

# 系统依赖（sqlite-vec 编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# torch CPU（~200MB，比 CUDA 版小很多）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# rsl_rl — PyPI 官方包 + tensorboard（Logger 依赖）
RUN pip install --no-cache-dir rsl-rl-lib tensorboard

# robotmem
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -e ".[dev,cjk]"

# 运行验证
CMD ["python", "-u", "examples/rsl_rl_demo/verify.py"]
