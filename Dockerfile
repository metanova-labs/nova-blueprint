# Runtime image for validator; code is mounted at /app.
# Deps are installed via uv in a dedicated venv; CUDA wheels installed explicitly.
FROM python:3.12-slim

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create a dedicated virtual environment and make it active
ENV VENV_DIR=/opt/venv
RUN uv venv ${VENV_DIR}
ENV VIRTUAL_ENV=${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:${PATH}"

# Pre-copy only lockfiles to leverage layer caching
WORKDIR /tmp/app
COPY pyproject.toml /tmp/app/pyproject.toml
COPY uv.lock /tmp/app/uv.lock

# Install locked dependencies into the active venv (no system site-packages)
RUN uv export --locked --no-dev -o /tmp/app/requirements.lock.txt \
    && uv pip install -r /tmp/app/requirements.lock.txt

# Install CUDA 12.6 torch and PyG wheels explicitly into the venv
RUN uv pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    && uv pip install torch-geometric==2.6.1 \
    && uv pip install -f https://data.pyg.org/whl/torch-2.7.0+cu126.html \
        pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv

# Working directory
WORKDIR /app

CMD ["bash"]

