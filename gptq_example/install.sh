#!/usr/bin/env bash
set -xe

pip install -U pip
pip install "vllm>=0.2.3"
pip install -U "setuptools==69.0.0"

pip install auto-gptq==0.7.1
pip install git+https://github.com/huggingface/optimum.git@3adbe7c75e3c41c1a3b945cf085e74ece7f8e19
pip install git+https://github.com/huggingface/transformers.git@b7fc2daf8b3fe783173c270d592073aabfb426c
pip install --upgrade accelerate==1.5.2
