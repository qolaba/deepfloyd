
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update --fix-missing

RUN apt install -y python3 python3-dev python3-pip

RUN pip3 install torch torchvision torchaudio

RUN pip install --upgrade diffusers~=0.16 transformers~=4.28 safetensors~=0.3 sentencepiece~=0.1 accelerate~=0.18 bitsandbytes~=0.38 torch~=2.0
RUN pip install huggingface_hub --upgrade
RUN pip install fastapi uvicorn locust git

RUN git clone https://github.com/qolaba/deepfloyd.git


WORKDIR /deepfloyd

ENTRYPOINT python3 df.py