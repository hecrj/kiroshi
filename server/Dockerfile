FROM rust:1.88 AS build

WORKDIR /kiroshi

COPY Cargo.toml /kiroshi/.
COPY Cargo.lock /kiroshi/.
COPY src/ /kiroshi/src

COPY server/src/ /kiroshi/server/src
COPY server/Cargo.toml /kiroshi/server/.

RUN cargo build --release --package kiroshi-server

FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="$PATH:/root/.local/bin"

RUN apt-get update && \
    apt install -y --no-install-recommends \
    software-properties-common \
    git \
    build-essential \
    libgl-dev \
    curl

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

# Copy server code
COPY server/. /kiroshi
COPY --from=build /kiroshi/target/release/kiroshi-server kiroshi

ENV YOLO_VERBOSE=false

ENTRYPOINT ["./kiroshi"]
