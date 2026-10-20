# Builder image for the DeepSeek-V4.1 toolchain (tools/dsv41/build.sh builds and tags one per flavor).
# --build-arg CUDA_FLAVOR=cu128|cu130 picks the base; both are pinned by digest so a moved tag cannot change the
# build. The CUDA devel images have gcc 11.4 but no git or CMake, and Ubuntu 22.04's apt CMake is 3.22 (Strata
# needs 3.24): CMake and Ninja come from pip, pinned so a rebuild configures the same. verify.sh runs the server's
# tests here, so jinja2 (with its markupsafe) and regex too, at requirements.txt's pins.
ARG CUDA_FLAVOR
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 AS base-cu128
FROM nvidia/cuda:13.0.1-devel-ubuntu22.04@sha256:93a8d207db5aaa6384f834a6bf70d417433f709e61b57a91e7cc99c16172f49c AS base-cu130
FROM base-${CUDA_FLAVOR}

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git python3 python3-pip \
    && pip3 install --no-cache-dir cmake==3.31.6 ninja==1.11.1.4 jinja2==3.1.6 markupsafe==3.0.3 regex==2026.9.10 \
    && rm -rf /var/lib/apt/lists/*
