FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV HOME=/root \
    PATH=/root/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        gnupg \
        ripgrep && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv python install 3.12 && \
    PYTHON_BIN="$(uv python find 3.12)" && \
    ln -sf "${PYTHON_BIN}" /usr/local/bin/python3.12 && \
    ln -sf "${PYTHON_BIN}" /usr/local/bin/python3 && \
    ln -sf "${PYTHON_BIN}" /usr/local/bin/python

RUN npm install -g @openai/codex @anthropic-ai/claude-code && \
    npm cache clean --force \
    claude install

RUN curl -fsS https://cursor.com/install | bash && \
    ln -sf /root/.local/bin/agent /root/.local/bin/cursor-agent

RUN mkdir -p /workspace/mmds /data

COPY scripts/mmds-dev-entrypoint.sh /usr/local/bin/mmds-dev-entrypoint
RUN chmod +x /usr/local/bin/mmds-dev-entrypoint

WORKDIR /workspace/mmds

ENTRYPOINT ["/usr/local/bin/mmds-dev-entrypoint"]
CMD ["sleep", "infinity"]
