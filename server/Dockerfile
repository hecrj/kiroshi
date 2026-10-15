FROM rust:1.88 AS build

WORKDIR /kiroshi

COPY Cargo.toml /kiroshi/.
COPY Cargo.lock /kiroshi/.
COPY src/ /kiroshi/src

COPY server/src/ /kiroshi/server/src
COPY server/Cargo.toml /kiroshi/server/.

RUN cargo build --release --package kiroshi-server

FROM nvidia/cuda:13.4.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="$PATH:/root/.local/bin"

RUN apt-get update && \
    apt install -y --no-install-recommends \
    software-properties-common \
    git \
    git-lfs \
    build-essential \
    libgl-dev \
    curl \
    python3-dev

# Enable Git LFS (model reference components are LFS-backed git repos)
RUN git lfs install

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv --version

WORKDIR /kiroshi

# Install Python dependencies
COPY server/pyproject.toml .
COPY server/uv.lock .

RUN uv sync

# Download weights
RUN mkdir weights

# Download upscaler models
RUN curl -L -o weights/RealESRGAN_x2.pth https://huggingface.co/hecrj/RealESRGAN/resolve/main/RealESRGAN_x2.pth?download=true
RUN curl -L -o weights/RealESRGAN_x2plus.pth https://huggingface.co/hecrj/RealESRGAN/resolve/main/RealESRGAN_x2plus.pth?download=true
RUN curl -L -o weights/RealESRGAN_x4.pth https://huggingface.co/hecrj/RealESRGAN/resolve/main/RealESRGAN_x4.pth?download=true
RUN curl -L -o weights/4x-UltraSharp.pth https://huggingface.co/hecrj/RealESRGAN/resolve/main/4x-UltraSharp.pth?download=true

# Download face and hand detection models
RUN curl -L -o weights/face_yolov8n.pt https://huggingface.co/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/face_yolov8n.pt?download=true
RUN curl -L -o weights/hand_yolov9c.pt https://huggingface.co/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/hand_yolov9c.pt?download=true

# Krea 2 reference components: the full reference dir (Qwen3-VL-4B text
# encoder, Qwen-Image VAE, tokenizer and pipeline config) is the hecrj/krea-2
# git repository (tracked as the server/templates/krea-2 submodule, with the
# safetensors via Git LFS). It is cloned only when built with
# --build-arg WITH_KREA2=1 (off by default; using a Krea 2 model without it
# fails with a clear error).
ARG WITH_KREA2=0
RUN if [ "$WITH_KREA2" = "1" ]; then \
    git clone --depth 1 https://huggingface.co/hecrj/krea-2 templates/krea-2; \
fi

# Anima reference components: the full reference dir (Qwen3-0.6B text
# encoder, Qwen-Image VAE, Qwen/T5 tokenizers) is the hecrj/anima git
# repository (tracked as the server/templates/anima submodule, with the
# safetensors via Git LFS). It is cloned only when built with
# --build-arg WITH_ANIMA=1 (off by default; using an Anima model without it
# fails with a clear error).
ARG WITH_ANIMA=0
RUN if [ "$WITH_ANIMA" = "1" ]; then \
    git clone --depth 1 https://huggingface.co/hecrj/anima templates/anima; \
fi

# Copy server code
COPY server/. /kiroshi
COPY --from=build /kiroshi/target/release/kiroshi-server kiroshi

ENV YOLO_VERBOSE=false

# Kernel caches live on named volumes that the host-side launcher
# (Server::run, src/server.rs) creates and mounts here:
# - /root/.nv         NVIDIA driver first-use kernel cache (cold = ~13s on
#                     the first generation; the driver cannot cache anywhere
#                     else, it always uses $HOME/.nv/ComputeCache)
# - /root/.triton     Triton kernel cache (Triton's own default location, so
#                     no env var is needed)
# - /root/.cache/torchinductor
#                     TorchInductor cache (its default, $TMPDIR/torchinductor_
#                     <user>, is ephemeral)
ENV TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor

ENTRYPOINT ["./kiroshi"]
