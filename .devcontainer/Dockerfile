ARG UV_VERSION=latest
ARG DEBIAN_VERSION=debian
ARG WORKSPACE_NAME_=workspace
ARG PROJECT_NAME_=${PROJECT_NAME}

FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv

FROM mcr.microsoft.com/vscode/devcontainers/base:$DEBIAN_VERSION
LABEL maintainer="datacation <dev@datacation.com>"

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends\
    procps \
    build-essential \
    curl \
    swig \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv --chown=vscode: /uv /uvx /bin/