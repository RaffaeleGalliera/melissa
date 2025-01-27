FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DISABLE_REQUIRE=true

# Install basic development tools and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        libgdbm-dev \
        libnss3-dev \
        python3-tk \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.11 from source
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && \
    tar -xzf Python-3.11.0.tgz && \
    cd Python-3.11.0 && \
    ./configure --enable-optimizations --with-ssl && \
    make altinstall

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip==23.2.1

# Set the working directory
WORKDIR /app

# Copy only requirements.txt to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application build
FROM base AS build

# Set environment variables for WandB and TensorBoard logs
ENV WANDB_DIR=/tmp/wandb_logs
ENV TENSORBOARD_LOGDIR=/tmp/tensorboard_logs

# Create log directories
RUN mkdir -p /tmp/wandb_logs /tmp/tensorboard_logs

COPY . /app

WORKDIR /app

RUN git config --global --add safe.directory /app

ENTRYPOINT ["/bin/bash"]
