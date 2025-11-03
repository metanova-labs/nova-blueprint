FROM python:3.12-slim

# Create non-root user
RUN addgroup --system appgroup && adduser --system --uid 10001 --ingroup appgroup mineruser

# Provide nova_ph2 code inside the image for imports like `import nova_ph2.*`
COPY libs/nova_ph2 /usr/local/lib/python3.12/site-packages/nova_ph2

# Ensure DB directory is writable by mineruser
RUN chown -R mineruser:appgroup /usr/local/lib/python3.12/site-packages/nova_ph2/combinatorial_db \
    && chmod -R 775 /usr/local/lib/python3.12/site-packages/nova_ph2/combinatorial_db

ENV PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    HF_HOME=/tmp
    
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && \
    python -m pip install torch-geometric==2.6.1 && \
    python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu126.html && \
    python -m pip install rdkit==2024.9.4 fair-esm==2.0.0 && \
    python -m pip install -r /usr/local/lib/python3.12/site-packages/nova_ph2/requirements/requirements.txt

ENV TORCH_HOME=/opt/torch_cache
RUN mkdir -p /opt/torch_cache && python -c "import esm; esm.pretrained.esm2_t33_650M_UR50D()"

# Drop privileges
USER mineruser

# Entrypoint: run /workspace/miner.py as a script
ENTRYPOINT ["python", "/workspace/miner.py"]


