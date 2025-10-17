# Base Image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set initial working directory
WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/ChenmienTan/RL2

# Set the working directory to the cloned repo
WORKDIR /workspace/RL2

RUN apt-get update && apt-get install -y python3-pip vim && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python packages
# NOTE: These were separate layers in the history but are combined here for better practice.
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    hydra-core \
    ninja \
    tqdm \
    transformers \
    peft \
    liger_kernel \
    wandb \
    torchdata \
    math_verify \
    "sglang[all]" \
    sglang_router \
    && pip install --no-cache-dir ring-flash-attn==0.1.8 || echo "Ring-flash-attn build failed, skipping..."

RUN pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation --no-deps

# Install additional system packages
RUN apt-get update && apt-get install -y numactl && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/outputs /workspace/logs

# Set environment variables for debugging and network configuration
ENV NCCL_DEBUG=INFO \
    NCCL_SOCKET_IFNAME=eth0 \
    TORCH_DISTRIBUTED_DEBUG=INFO

# Set the default command to start a bash shell
CMD ["/bin/bash"]