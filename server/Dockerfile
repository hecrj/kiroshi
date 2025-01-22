FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt install -y --no-install-recommends \
    software-properties-common \
    git \
    build-essential \
    libgl-dev

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11-dev python3.11-venv

RUN python3.11 --version

WORKDIR /kiroshi

# Install Python dependencies
COPY requirements.txt .
RUN python3.11 -m venv .env
RUN .env/bin/pip install -r requirements.txt

RUN mkdir weights

RUN apt-get update && apt-get install -y curl

# Download upscaler models
RUN curl -L -o weights/RealESRGAN_x2.pth https://huggingface.co/ai-forever/Real-ESRGAN/resolve/a86fc6182b4650b4459cb1ddcb0a0d1ec86bf3b0/RealESRGAN_x2.pth?download=true
RUN curl -L -o weights/RealESRGAN_x4.pth https://huggingface.co/ai-forever/Real-ESRGAN/resolve/a86fc6182b4650b4459cb1ddcb0a0d1ec86bf3b0/RealESRGAN_x4.pth?download=true

# Download face and hand detection models
RUN curl -L -o weights/face_yolov8n.pt https://huggingface.co/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/face_yolov8n.pt?download=true
RUN curl -L -o weights/hand_yolov9c.pt https://huggingface.co/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/hand_yolov9c.pt?download=true

# Copy server code
COPY . /kiroshi

ENTRYPOINT [".env/bin/python", "main.py"]
