# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# Create non-root user
RUN addgroup --system appgroup && adduser --system --uid 10001 --ingroup appgroup mineruser

# Provide nova_miner code inside the image for imports like `import nova_miner.*`
COPY libs/nova_miner /usr/local/lib/python3.12/site-packages/nova_miner

ADD https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/sanqi/molecules.sqlite /usr/local/lib/python3.12/site-packages/nova_miner/combinatorial_db/molecules.sqlite

# Ensure DB directory is writable by mineruser
RUN chown -R mineruser:appgroup /usr/local/lib/python3.12/site-packages/nova_miner/combinatorial_db \
    && chmod -R 775 /usr/local/lib/python3.12/site-packages/nova_miner/combinatorial_db

ENV PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /usr/local/lib/python3.12/site-packages/nova_miner/requirements/sandbox.txt

# Drop privileges
USER mineruser

# Entrypoint: run /workspace/miner.py as a script
ENTRYPOINT ["python", "/workspace/miner.py"]
