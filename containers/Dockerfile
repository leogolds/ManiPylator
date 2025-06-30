# Derived from Genesis Dockerfile: https://github.com/Genesis-Embodied-AI/Genesis/blob/2235ce0bf2c684b3aa05e16a6a6c4984ce783302/docker/Dockerfile

# # ===============================================================
# # Stage 1: Build LuisaRender
# # ===============================================================
# FROM pytorch/pytorch:2.5.1-cuda${CUDA_VERSION}-cudnn9-devel AS builder

# ENV DEBIAN_FRONTEND=noninteractive
# ARG PYTHON_VERSION=3.11

# # Install necessary packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     manpages-dev \
#     libvulkan-dev \
#     zlib1g-dev \
#     xorg-dev libglu1-mesa-dev \
#     libsnappy-dev \
#     software-properties-common \
#     git \
#     curl \
#     wget
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
#     apt update && \
#     apt install -y --no-install-recommends \
#     gcc-11 \
#     g++-11 \
#     gcc-11 g++-11 patchelf && \
#     rm -rf /var/lib/apt/lists/*

# # Set GCC-11 and G++-11 as the default
# RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 && \
#     update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# # Install Rust for build requirements
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# RUN pip install "pybind11[global]"

# # Install CMake
# RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.0-rc2/cmake-3.31.0-rc2-linux-x86_64.sh && \
#     chmod +x cmake-3.31.0-rc2-linux-x86_64.sh && \
#     ./cmake-3.31.0-rc2-linux-x86_64.sh --skip-license --prefix=/usr/local && \
#     rm cmake-3.31.0-rc2-linux-x86_64.sh

# # Build LuisaRender
# WORKDIR /workspace
# RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
#     cd Genesis && \
#     git submodule update --init --recursive
# COPY build_luisa.sh /workspace/build_luisa.sh
# RUN chmod +x ./build_luisa.sh && ./build_luisa.sh ${PYTHON_VERSION}

# ===============================================================
# Stage 2: Runtime Environment
# ===============================================================
ARG CUDA_VERSION=12.4
FROM pytorch/pytorch:2.5.1-cuda${CUDA_VERSION}-cudnn9-devel

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    curl \
    wget \
    bash-completion \
    libgl1 \
    libgl1-mesa-glx \
    libegl-dev \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
    libvulkan-dev \
    libgles2 \
    libglvnd0 \
    libglx0 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --------------------------- Genesis ----------------------------
RUN pip install --no-cache-dir open3d
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
    cd Genesis && \
    pip install . && \
    pip install --no-cache-dir PyOpenGL==3.1.5

# ------------------------ Motion planning -----------------------
RUN PYTHON_MAJOR_MINOR=$(echo ${PYTHON_VERSION} | tr -d '.') && \
    wget https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-manylinux_2_28_x86_64.whl && \
    pip install ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-manylinux_2_28_x86_64.whl && \
    rm ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-manylinux_2_28_x86_64.whl

# ----------------------- Personal deps ---------------------------
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash"]
