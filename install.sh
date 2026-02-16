#!/bin/bash


set -e  

pip uninstall wandb -y
pip install byted-wandb==0.13.90 -i https://bytedpypi.byted.org/simple
pip install math-verify mathruler
pip install qwen_vl_utils==0.0.14
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

pip install sglang==0.5.5.post3
pip install transformers==4.55.3
pip install httpx==0.23
pip install pybase64
pip install protobuf==3.20.3
pip install flash-attn==2.8.1 --no-build-isolation

