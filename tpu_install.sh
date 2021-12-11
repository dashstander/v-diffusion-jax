#!/usr/bin/bash
pip3 install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install git+https://github.com/openai/CLIP.git
git clone https://github.com/kingoflolz/CLIP_JAX.git
pip3 install -r CLIP_JAX/requirements.txt
pip3 install -e CLIP_JAX/
pip3 install -r requirements.txt
pip3 install -e .
