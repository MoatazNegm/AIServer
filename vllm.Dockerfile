# vLLM container for this host's RTX A4500
# Network-restricted build:
#   - Base: nvcr.io/nvidia/cuda:12.4.1-base-ubi9 (cached locally or pulled fresh
#     from nvcr.io, which is the only reliable registry on this host)
#   - System packages: Python 3.11, gcc/g++/make (Triton JIT-needs gcc at runtime)
#   - vLLM 0.10.0 from PyPI via Tsinghua mirror (pypi.org stalls on large wheels)
#   - cuDNN/cuBLAS/cuSPARSE/cuSPARSE-Lt/etc come as transitive deps of vllm/torch
#     and are installed via pip
#
# Build:    docker build -f vllm.Dockerfile -t vllm-a4500:latest .
# Run:      ./start-vllm.sh Qwen/Qwen2.5-7B-Instruct
# Stop:     docker stop vllm-server

FROM nvcr.io/nvidia/cuda:12.4.1-base-ubi9

# System packages
#   - python3.11: Python runtime
#   - gcc/gcc-c++/make: REQUIRED at vllm/torch runtime because Triton JIT-compiles
#     CUDA kernels and shells out to a C compiler. Without it vllm fails with
#     "Failed to find C compiler" the first time you load a model.
RUN dnf install -y \
        python3.11 \
        python3.11-pip \
        python3.11-devel \
        git \
        ca-certificates \
        gcc \
        gcc-c++ \
        make \
        && dnf clean all \
        && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
        && ln -sf /usr/bin/pip3.11   /usr/bin/pip3

# Tsinghua PyPI mirror (PyPI.org stalls on large wheels from this host)
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# vLLM 0.10.0 calls `tokenizer.all_special_tokens_extended`, which was removed
# in transformers 5.x. Pin transformers<5 before vllm so the resolver picks
# a 4.x version (transformers 4.55.x is the last 4-line).
ARG VLLM_VERSION=0.10.0
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir "transformers<5" "vllm==${VLLM_VERSION}"

# Discover every pip-installed lib directory under site-packages and register
# them with ldconfig. Most nvidia-* packages install to nvidia/<name>/lib,
# but nvidia-cusparselt-cu12 installs to cusparselt/lib (no nvidia/ prefix).
RUN NVIDIA_LIB_PATHS=$(find /usr/local/lib/python3.11/site-packages -mindepth 2 -maxdepth 3 -type d -name lib | sort -u | paste -sd:) \
    && echo "Discovered lib paths:" \
    && echo "$NVIDIA_LIB_PATHS" | tr ':' '\n' \
    && printf '%s\n' $NVIDIA_LIB_PATHS > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig \
    && ldconfig -p | grep -E 'libcudnn|libcusparseLt|libcublas|libnccl' | head -10

# Bake LD_LIBRARY_PATH into the image so torch/vllm can dlopen libcudnn.so.9
# at RUNTIME. Docker ENV doesn't see $NVIDIA_LIB_PATHS from earlier RUNs, so
# the full path list is duplicated here. If vllm is bumped and the list
# changes, run the find again and paste the new list.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufile/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvtx/lib:/usr/local/lib/python3.11/site-packages/cusparselt/lib:/usr/local/lib/python3.11/site-packages/openai/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Sanity check at build time — fails the build (not the run) if imports break.
# Note: torch.cuda.is_available() is False during `docker build` (no GPU in
# build context); the real GPU test happens at `docker run --gpus all`.
RUN python3 - <<'PY'
import torch, vllm
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('vllm', vllm.__version__)
# No assert — GPU is not visible during `docker build`.
PY

# Default entrypoint runs the upstream vllm OpenAI-compatible server.
# The CMD below is the canonical baseline: 32k context, fp16, tool calling
# on (hermes parser), bind on all interfaces. start-vllm.sh overrides
# --model and CMD args at run time; everything else is the recommended
# baseline.
#
# This image can ALSO serve as the gateway container. To use it as a
# gateway, override the entrypoint and CMD, e.g.:
#   docker run ... --entrypoint python3 \
#     vllm-a4500:latest -m uvicorn app:app --app-dir /app ...
# The /app/app.py below is the auth-enforcing, OpenAI-compatible gateway
# with smart max_tokens capping (MODEL_MAX_CONTEXT / MAX_OUTPUT_TOKENS_HARD_CAP).
COPY gateway/app.py /app/app.py

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--max-model-len", "32768", "--gpu-memory-utilization", "0.9", "--dtype", "float16", "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--host", "0.0.0.0", "--port", "8000"]
