# Docker image for nova-blueprint; code runs from /app. CUDA deps installed via uv.
FROM python:3.12-slim AS base

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install a static Docker CLI to ensure 'docker' is available 
ENV DOCKER_CLI_VERSION=24.0.9
RUN curl -fsSL -o /tmp/docker.tgz https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz \
    && tar -xz -C /usr/local/bin --strip-components=1 -f /tmp/docker.tgz docker/docker \
    && rm /tmp/docker.tgz \
    && chmod +x /usr/local/bin/docker \
    && /usr/local/bin/docker --version || true

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create a dedicated virtual environment and make it active
ENV VENV_DIR=/opt/venv
RUN uv venv ${VENV_DIR}
ENV VIRTUAL_ENV=${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:${PATH}"

# Install locked dependencies into the active venv
WORKDIR /tmp/app
COPY pyproject.toml /tmp/app/pyproject.toml
COPY uv.lock /tmp/app/uv.lock
RUN uv export --locked --no-dev -o /tmp/app/requirements.lock.txt \
    && uv pip install -r /tmp/app/requirements.lock.txt

# Install CUDA 12.6 torch and PyG wheels into the venv
RUN uv pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    && uv pip install torch-geometric==2.6.1 \
    && uv pip install -f https://data.pyg.org/whl/torch-2.7.0+cu126.html \
        pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv

# Default working directory
WORKDIR /app

# Final stage with app code baked in
FROM base AS final
WORKDIR /app
COPY . /app
CMD ["python", "neurons/validator/scheduler.py"]

